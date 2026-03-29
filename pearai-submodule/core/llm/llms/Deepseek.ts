import { CompletionOptions, LLMOptions, ModelProvider } from "../../index.js";
import { streamSse } from "../stream.js";
import { osModelsEditPrompt } from "../templates/edit.js";
import OpenAI from "./OpenAI.js";

// DeepSeek reasoning models (R1 series) use different handling
const DEEPSEEK_REASONING_MODELS = [
  "deepseek-reasoner",
  "deepseek-r1",
];

class Deepseek extends OpenAI {
  static providerName: ModelProvider = "deepseek";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.deepseek.com/",
    // Updated to deepseek-chat (maps to DeepSeek-V3)
    model: "deepseek-chat",
    promptTemplates: {
      edit: osModelsEditPrompt,
    },
    useLegacyCompletionsEndpoint: false,
  };

  protected maxStopWords: number | undefined = 16;

  private _isReasoningModel(model: string): boolean {
    return DEEPSEEK_REASONING_MODELS.some((m) => model.startsWith(m));
  }

  supportsImages(): boolean {
    // deepseek-chat (V3) doesn't support images natively
    return false;
  }

  supportsFim(): boolean {
    // Only deepseek-coder supports FIM
    return this.model?.startsWith("deepseek-coder") ?? false;
  }

  protected async *_streamChat(
    messages: any[],
    options: CompletionOptions,
  ): AsyncGenerator<any> {
    // For reasoning models, prefix reasoning content is separate
    if (this._isReasoningModel(options.model)) {
      // Yield thinking prefix if present in the response
      for await (const chunk of super._streamChat(messages, options)) {
        yield chunk;
      }
      return;
    }
    yield* super._streamChat(messages, options);
  }

  async *_streamFim(
    prefix: string,
    suffix: string,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const endpoint = new URL("beta/completions", this.apiBase);
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
        Authorization: `Bearer ${this.apiKey}`,
      },
    });
    for await (const chunk of streamSse(resp)) {
      yield chunk.choices[0].text;
    }
  }
}

export default Deepseek;
