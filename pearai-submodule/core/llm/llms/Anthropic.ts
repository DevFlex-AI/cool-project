import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  ModelProvider,
} from "../../index.js";
import { stripImages } from "../images.js";
import { BaseLLM } from "../index.js";
import { streamSse } from "../stream.js";

// Models that support extended thinking / reasoning
const THINKING_MODELS = [
  "claude-3-7-sonnet-20250219",
  "claude-3-7-sonnet-latest",
];

// Models that use a higher output token limit
const HIGH_OUTPUT_TOKEN_MODELS = [
  "claude-3-7-sonnet-20250219",
  "claude-3-7-sonnet-latest",
  "claude-3-5-sonnet-20241022",
  "claude-3-5-haiku-20241022",
];

class Anthropic extends BaseLLM {
  static providerName: ModelProvider = "anthropic";
  static defaultOptions: Partial<LLMOptions> = {
    // Updated to latest stable Claude model
    model: "claude-3-5-sonnet-20241022",
    contextLength: 200_000,
    completionOptions: {
      model: "claude-3-5-sonnet-20241022",
      maxTokens: 8096,
    },
    apiBase: "https://api.anthropic.com/v1/",
  };

  private _isThinkingModel(model: string): boolean {
    return THINKING_MODELS.some((m) => model.startsWith(m));
  }

  private _convertArgs(options: CompletionOptions) {
    const isHighOutput = HIGH_OUTPUT_TOKEN_MODELS.some((m) =>
      options.model.startsWith(m),
    );

    const finalOptions: Record<string, any> = {
      top_k: options.topK,
      top_p: options.topP,
      temperature: options.temperature,
      max_tokens: options.maxTokens ?? (isHighOutput ? 8096 : 4096),
      model: options.model === "claude-2" ? "claude-2.1" : options.model,
      stop_sequences: options.stop?.filter((x) => x.trim() !== ""),
      stream: options.stream ?? true,
    };

    // Enable extended thinking for supported models
    if (this._isThinkingModel(options.model) && options.temperature === 1) {
      finalOptions.thinking = {
        type: "enabled",
        budget_tokens: Math.min(options.maxTokens ?? 8096, 10000),
      };
      // Thinking requires temp=1
      finalOptions.temperature = 1;
    }

    return finalOptions;
  }

  private _convertMessages(msgs: ChatMessage[]): any[] {
    const messages = msgs
      .filter((m) => m.role !== "system")
      .map((message) => {
        if (typeof message.content === "string") {
          return message;
        }
        return {
          ...message,
          content: message.content.map((part) => {
            if (part.type === "text") {
              return part;
            }
            return {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/jpeg",
                data: part.imageUrl?.url.split(",")[1],
              },
            };
          }),
        };
      });
    return messages;
  }

  protected async *_streamComplete(
    prompt: string,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const messages = [{ role: "user" as const, content: prompt }];
    for await (const update of this._streamChat(messages, options)) {
      yield stripImages(update.content);
    }
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const response = await this.fetch(new URL("messages", this.apiBase), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "anthropic-version": "2023-06-01",
        // Required for extended thinking and latest features
        "anthropic-beta": "interleaved-thinking-2025-05-14",
        "x-api-key": this.apiKey as string,
      },
      body: JSON.stringify({
        ...this._convertArgs(options),
        messages: this._convertMessages(messages),
        system: this.systemMessage,
      }),
    });

    if (options.stream === false) {
      const data = await response.json();
      // Handle both text and thinking blocks
      const textContent = Array.isArray(data.content)
        ? data.content
            .filter((b: any) => b.type === "text")
            .map((b: any) => b.text)
            .join("")
        : data.content[0].text;
      yield { role: "assistant", content: textContent };
      return;
    }

    for await (const value of streamSse(response)) {
      if (value.type === "content_block_delta") {
        if (value.delta?.type === "text_delta" && value.delta?.text) {
          yield { role: "assistant", content: value.delta.text };
        }
        // Skip thinking deltas (they are internal reasoning, not shown to user)
      } else if (value.delta?.text) {
        // Legacy fallback
        yield { role: "assistant", content: value.delta.text };
      }
    }
  }

  supportsFim(): boolean {
    return false;
  }
}

export default Anthropic;
