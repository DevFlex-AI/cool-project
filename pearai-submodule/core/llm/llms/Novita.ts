import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * Novita AI — serverless GPU inference
 * Models: llama-3, mistral, deepseek, qwen, and more
 */
class Novita extends OpenAI {
  static providerName: ModelProvider = "novita" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.novita.ai/v3/openai/",
    model: "meta-llama/llama-3.3-70b-instruct",
  };

  supportsCompletions(): boolean {
    return false;
  }
}

export default Novita;
