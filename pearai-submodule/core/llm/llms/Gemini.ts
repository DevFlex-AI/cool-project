import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  MessagePart,
  ModelProvider,
} from "../../index.js";
import { stripImages } from "../images.js";
import { BaseLLM } from "../index.js";
import { streamResponse } from "../stream.js";

// Models with large context windows (up to 2M tokens)
const LARGE_CONTEXT_MODELS = [
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "gemini-2.0-pro",
  "gemini-1.5-pro",
  "gemini-1.5-flash",
];

// Thinking/reasoning capable models
const THINKING_MODELS = [
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "gemini-2.0-flash-thinking",
];

class Gemini extends BaseLLM {
  static providerName: ModelProvider = "gemini";

  static defaultOptions: Partial<LLMOptions> = {
    // Updated to latest stable Gemini model
    model: "gemini-2.0-flash",
    contextLength: 1_048_576,
    apiBase: "https://generativelanguage.googleapis.com/v1beta/",
  };

  private _isThinkingModel(model: string): boolean {
    return THINKING_MODELS.some((m) => model.startsWith(m));
  }

  private _getContextLength(model: string): number {
    if (LARGE_CONTEXT_MODELS.some((m) => model.startsWith(m))) {
      return 2_097_152; // 2M context
    }
    return 1_048_576; // 1M context default
  }

  // Function to convert completion options to Gemini format
  private _convertArgs(options: CompletionOptions) {
    const finalOptions: any = {};

    if (options.topK) {
      finalOptions.topK = options.topK;
    }
    if (options.topP) {
      finalOptions.topP = options.topP;
    }
    if (options.temperature !== undefined && options.temperature !== null) {
      finalOptions.temperature = options.temperature;
    }
    if (options.maxTokens) {
      finalOptions.maxOutputTokens = options.maxTokens;
    }
    if (options.stop) {
      finalOptions.stopSequences = options.stop.filter((x) => x.trim() !== "");
    }

    const config: any = { generationConfig: finalOptions };

    // Enable thinking for supported models
    if (this._isThinkingModel(options.model)) {
      config.generationConfig.thinkingConfig = {
        thinkingBudget: -1, // Dynamic thinking budget
      };
    }

    return config;
  }

  protected async *_streamComplete(
    prompt: string,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    for await (const message of this._streamChat(
      [{ content: prompt, role: "user" }],
      options,
    )) {
      yield stripImages(message.content);
    }
  }

  private removeSystemMessage(messages: ChatMessage[]) {
    const msgs = [...messages];

    if (msgs[0]?.role === "system") {
      const sysMsg = msgs.shift()?.content;
      // @ts-ignore
      if (msgs[0]?.role === "user") {
        msgs[0].content = `System message - follow these instructions in every response: ${sysMsg}\n\n---\n\n${msgs[0].content}`;
      }
    }

    return msgs;
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const apiBase =
      this.apiBase ||
      Gemini.defaultOptions?.apiBase ||
      "https://generativelanguage.googleapis.com/v1beta/";
    const isV1API = apiBase.includes("/v1/");

    const convertedMsgs = isV1API
      ? this.removeSystemMessage(messages)
      : messages;

    if (options.model.includes("gemini")) {
      for await (const message of this.streamChatGemini(
        convertedMsgs,
        options,
      )) {
        yield message;
      }
    } else {
      for await (const message of this.streamChatBison(
        convertedMsgs,
        options,
      )) {
        yield message;
      }
    }
  }

  private _continuePartToGeminiPart(part: MessagePart) {
    return part.type === "text"
      ? {
          text: part.text,
        }
      : {
          inlineData: {
            mimeType: "image/jpeg",
            data: part.imageUrl?.url.split(",")[1],
          },
        };
  }

  private async *streamChatGemini(
    messages: ChatMessage[],
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const apiBase =
      this.apiBase ||
      Gemini.defaultOptions?.apiBase ||
      "https://generativelanguage.googleapis.com/v1beta/";
    const isV1API = apiBase.includes("/v1/");

    const apiURL = new URL(
      `models/${options.model}:streamGenerateContent?key=${this.apiKey}&alt=sse`,
      apiBase,
    );

    const contents = messages
      .map((msg) => {
        if (msg.role === "system" && !isV1API) {
          return null;
        }
        return {
          role: msg.role === "assistant" ? "model" : "user",
          parts:
            typeof msg.content === "string"
              ? [{ text: msg.content }]
              : msg.content.map(this._continuePartToGeminiPart),
        };
      })
      .filter((c) => c !== null);

    const body: any = {
      ...this._convertArgs(options),
      contents,
      ...(this.systemMessage &&
        !isV1API && {
          systemInstruction: { parts: [{ text: this.systemMessage }] },
        }),
    };

    const response = await this.fetch(apiURL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    let buffer = "";
    for await (const chunk of streamResponse(response)) {
      buffer += chunk;
      if (buffer.startsWith("[")) {
        buffer = buffer.slice(1);
      }
      if (buffer.endsWith("]")) {
        buffer = buffer.slice(0, -1);
      }
      if (buffer.startsWith(",")) {
        buffer = buffer.slice(1);
      }

      const parts = buffer.split("\n,");

      let foundIncomplete = false;
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        let data;
        try {
          data = JSON.parse(part);
        } catch (e) {
          foundIncomplete = true;
          continue;
        }
        if (data.error) {
          throw new Error(data.error.message);
        }
        // Handle multiple parts in response (text + thinking)
        if (data?.candidates?.[0]?.content?.parts) {
          for (const responsePart of data.candidates[0].content.parts) {
            // Only yield text parts, skip thought/thinking parts
            if (responsePart.text && !responsePart.thought) {
              yield {
                role: "assistant",
                content: responsePart.text,
              };
            }
          }
        } else {
          console.warn("Unexpected response format:", data);
        }
      }
      if (foundIncomplete) {
        buffer = parts[parts.length - 1];
      } else {
        buffer = "";
      }
    }
  }

  private async *streamChatBison(
    messages: ChatMessage[],
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const msgList = [];
    for (const message of messages) {
      msgList.push({ content: message.content });
    }

    const apiURL = new URL(
      `models/${options.model}:generateMessage?key=${this.apiKey}`,
      this.apiBase,
    );
    const body = { prompt: { messages: msgList } };
    const response = await this.fetch(apiURL, {
      method: "POST",
      body: JSON.stringify(body),
    });
    const data = await response.json();
    yield { role: "assistant", content: data.candidates[0].content };
  }

  supportsImages(): boolean {
    // All modern Gemini models support images
    return true;
  }
}

async function delay(seconds: number) {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

export default Gemini;
