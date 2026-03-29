import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  ModelProvider,
} from "../../index.js";
import { stripImages } from "../images.js";
import { BaseLLM } from "../index.js";
import { streamSse } from "../stream.js";

const NON_CHAT_MODELS = [
  "text-davinci-002",
  "text-davinci-003",
  "code-davinci-002",
  "text-ada-001",
  "text-babbage-001",
  "text-curie-001",
  "davinci",
  "curie",
  "babbage",
  "ada",
];

const CHAT_ONLY_MODELS = [
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-0613",
  "gpt-3.5-turbo-16k",
  "gpt-4",
  "gpt-4-turbo",
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4o-2024-11-20",
  "gpt-4o-2024-08-06",
  "gpt-4-5",
  "gpt-35-turbo-16k",
  "gpt-35-turbo-0613",
  "gpt-35-turbo",
  "gpt-4-32k",
  "gpt-4-turbo-preview",
  "gpt-4-vision",
  "gpt-4-0125-preview",
  "gpt-4-1106-preview",
];

// O-series reasoning models (require special handling)
const O_SERIES_MODELS = [
  "o1",
  "o1-mini",
  "o1-preview",
  "o3",
  "o3-mini",
  "o4-mini",
];

// Models that use max_completion_tokens instead of max_tokens
const MAX_COMPLETION_TOKEN_MODELS = [...O_SERIES_MODELS, "gpt-4-5"];

class OpenAI extends BaseLLM {
  public useLegacyCompletionsEndpoint: boolean | undefined = undefined;

  protected maxStopWords: number | undefined = undefined;

  constructor(options: LLMOptions) {
    super(options);
    this.useLegacyCompletionsEndpoint = options.useLegacyCompletionsEndpoint;
    this.apiVersion = options.apiVersion ?? "2024-02-01";
  }

  static providerName: ModelProvider = "openai";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.openai.com/v1/",
    model: "gpt-4o",
  };

  protected _convertMessage(message: ChatMessage) {
    if (typeof message.content === "string") {
      return message;
    } else if (!message.content.some((item) => item.type !== "text")) {
      return {
        ...message,
        content: message.content.map((item) => item.text).join(""),
      };
    }

    const parts = message.content.map((part) => {
      const msg: any = {
        type: part.type,
        text: part.text,
      };
      if (part.type === "imageUrl") {
        msg.image_url = { ...part.imageUrl, detail: "low" };
        msg.type = "image_url";
      }
      return msg;
    });
    return {
      ...message,
      content: parts,
    };
  }

  protected _convertModelName(model: string): string {
    return model;
  }

  private _isOSeriesModel(model?: string): boolean {
    if (!model) return false;
    return O_SERIES_MODELS.some(
      (m) => model === m || model.startsWith(m + "-"),
    );
  }

  private _isO1LegacyModel(model?: string): boolean {
    // o1-preview and o1-mini have extra restrictions (no streaming, no system msg)
    return !!(
      model &&
      (model.startsWith("o1-preview") || model.startsWith("o1-mini"))
    );
  }

  protected _convertArgs(options: any, messages: ChatMessage[]) {
    const url = new URL(this.apiBase!);
    const isOSeries = this._isOSeriesModel(options.model);
    const isO1Legacy = this._isO1LegacyModel(options.model);

    let convertedMessages = messages.map(this._convertMessage);

    // O1-legacy: remove system messages (not supported)
    if (isO1Legacy) {
      convertedMessages = convertedMessages.filter(
        (m: any) => m?.role !== "system",
      );
    } else if (isOSeries) {
      // Newer o-series: convert system -> developer role
      convertedMessages = convertedMessages.map((m: any) =>
        m?.role === "system" ? { ...m, role: "developer" } : m,
      );
    }

    const finalOptions: any = {
      messages: convertedMessages,
      model: this._convertModelName(options.model),
      temperature: isOSeries ? undefined : options.temperature, // o-series ignores temperature
      top_p: isOSeries ? undefined : options.topP,
      frequency_penalty: isOSeries ? undefined : options.frequencyPenalty,
      presence_penalty: isOSeries ? undefined : options.presencePenalty,
      stream: isO1Legacy ? false : (options.stream ?? true), // o1-legacy no streaming
      stop:
        this.maxStopWords !== undefined
          ? options.stop?.slice(0, this.maxStopWords)
          : url.host === "api.deepseek.com"
            ? options.stop?.slice(0, 16)
            : url.port === "1337" ||
                url.host === "api.openai.com" ||
                url.host === "api.groq.com" ||
                this.apiType === "azure"
              ? options.stop?.slice(0, 4)
              : options.stop,
    };

    // Use max_completion_tokens for o-series and gpt-4.5+
    if (MAX_COMPLETION_TOKEN_MODELS.some((m) => options.model.startsWith(m))) {
      finalOptions.max_completion_tokens = options.maxTokens;
    } else {
      finalOptions.max_tokens = options.maxTokens;
    }

    // Reasoning effort for o-series (o3, o4-mini support this)
    if (isOSeries && !isO1Legacy && options.reasoningEffort) {
      finalOptions.reasoning_effort = options.reasoningEffort;
    }

    return finalOptions;
  }

  protected _getHeaders() {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.apiKey}`,
      "api-key": this.apiKey ?? "", // For Azure
    };
  }

  protected async _complete(
    prompt: string,
    options: CompletionOptions,
  ): Promise<string> {
    let completion = "";
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      options,
    )) {
      completion += chunk.content;
    }

    return completion;
  }

  private _getEndpoint(
    endpoint: "chat/completions" | "completions" | "models",
  ) {
    if (this.apiType === "azure") {
      return new URL(
        `openai/deployments/${this.engine}/${endpoint}?api-version=${this.apiVersion}`,
        this.apiBase,
      );
    }
    if (!this.apiBase) {
      throw new Error(
        "No API base URL provided. Please set the 'apiBase' option in config.json",
      );
    }

    return new URL(endpoint, this.apiBase);
  }

  protected async *_streamComplete(
    prompt: string,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      options,
    )) {
      yield stripImages(chunk.content);
    }
  }

  protected async *_legacystreamComplete(
    prompt: string,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const args: any = this._convertArgs(options, []);
    args.prompt = prompt;
    args.messages = undefined;

    const response = await this.fetch(this._getEndpoint("completions"), {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify({
        ...args,
        stream: true,
      }),
    });

    for await (const value of streamSse(response)) {
      if (value.choices?.[0]?.text && value.finish_reason !== "eos") {
        yield value.choices[0].text;
      }
    }
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    if (
      !CHAT_ONLY_MODELS.includes(options.model) &&
      this.supportsCompletions() &&
      (NON_CHAT_MODELS.includes(options.model) ||
        this.useLegacyCompletionsEndpoint ||
        options.raw)
    ) {
      for await (const content of this._legacystreamComplete(
        stripImages(messages[messages.length - 1]?.content || ""),
        options,
      )) {
        yield {
          role: "assistant",
          content,
        };
      }
      return;
    }

    const body = this._convertArgs(options, messages);
    // Empty messages cause errors in some servers
    body.messages = body.messages.map((m: any) => ({
      ...m,
      content: m.content === "" ? " " : m.content,
    })) as any;

    const response = await this.fetch(this._getEndpoint("chat/completions"), {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify(body),
    });

    // Handle non-streaming response (o1-legacy models)
    if (body.stream === false) {
      const data = await response.json();
      if (data.error) {
        throw new Error(
          `OpenAI API error: ${data.error.message || JSON.stringify(data.error)}`,
        );
      }
      yield data.choices[0].message;
      return;
    }

    for await (const value of streamSse(response)) {
      if (value.choices?.[0]?.delta?.content) {
        yield value.choices[0].delta;
      }
      // Handle reasoning tokens (o-series thinking output) — skip to user
      if (value.choices?.[0]?.delta?.reasoning_content) {
        // Reasoning content is internal — don't yield to chat
        continue;
      }
    }
  }

  async *_streamFim(
    prefix: string,
    suffix: string,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const endpoint = new URL("fim/completions", this.apiBase);
    const resp = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({
        model: options.model,
        prompt: prefix,
        suffix,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
        top_p: options.topP,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        stop: options.stop,
        stream: true,
      }),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "x-api-key": this.apiKey ?? "",
        Authorization: `Bearer ${this.apiKey}`,
      },
    });
    for await (const chunk of streamSse(resp)) {
      yield chunk.choices[0].delta.content;
    }
  }

  async listModels(): Promise<string[]> {
    const response = await this.fetch(this._getEndpoint("models"), {
      method: "GET",
      headers: this._getHeaders(),
    });

    const data = await response.json();
    return data.data.map((m: any) => m.id);
  }

  supportsImages(): boolean {
    const model = this.model;
    if (!model) return false;
    // GPT-4V, GPT-4o, and o-series with vision support images
    return (
      model.includes("gpt-4") ||
      model.includes("gpt-4o") ||
      model.startsWith("o1") ||
      model.startsWith("o3") ||
      model.startsWith("o4")
    );
  }
}

export default OpenAI;
