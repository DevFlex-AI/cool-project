import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * Cerebras — ultra-fast inference via the Wafer-Scale Engine
 * Models: llama3.1-8b, llama3.1-70b, llama3.3-70b, llama-4-scout-17b
 */
class Cerebras extends OpenAI {
  static providerName: ModelProvider = "cerebras" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.cerebras.ai/v1/",
    model: "llama-3.3-70b",
  };

  maxStopWords: number | undefined = 4;

  supportsCompletions(): boolean {
    return false;
  }
}

export default Cerebras;
