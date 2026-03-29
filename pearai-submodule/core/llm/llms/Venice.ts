import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * Venice AI — privacy-focused AI inference
 * Models: llama-3.3-70b, mistral-31-24b, qwen-2.5-vl, etc.
 */
class Venice extends OpenAI {
  static providerName: ModelProvider = "venice" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.venice.ai/api/v1/",
    model: "llama-3.3-70b",
  };

  supportsCompletions(): boolean {
    return false;
  }
}

export default Venice;
