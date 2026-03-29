import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * SambaNova Cloud — high-throughput inference
 * Models: llama3.3-70b, llama3.1-405b, deepseek-r1, deepseek-v3, qwen3-32b
 */
class SambaNova extends OpenAI {
  static providerName: ModelProvider = "sambanova" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.sambanova.ai/v1/",
    model: "llama3.3-70b",
  };

  private static MODEL_IDS: { [name: string]: string } = {
    "llama3.3-70b": "Meta-Llama-3.3-70B-Instruct",
    "llama3.1-8b": "Meta-Llama-3.1-8B-Instruct",
    "llama3.1-70b": "Meta-Llama-3.1-70B-Instruct",
    "llama3.1-405b": "Meta-Llama-3.1-405B-Instruct",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3-0324",
    "qwen3-32b": "Qwen3-32B",
  };

  protected _convertModelName(model: string): string {
    return SambaNova.MODEL_IDS[model] || model;
  }

  supportsCompletions(): boolean {
    return false;
  }
}

export default SambaNova;
