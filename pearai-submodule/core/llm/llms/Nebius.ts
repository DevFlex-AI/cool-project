import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * Nebius AI Studio — fast GPU inference
 * Models: deepseek-v3, deepseek-r1, qwen2.5-coder-32b, llama3.1-70b, etc.
 */
class Nebius extends OpenAI {
  static providerName: ModelProvider = "nebius" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.studio.nebius.ai/v1/",
    model: "deepseek/deepseek_v3",
  };

  private static MODEL_IDS: { [name: string]: string } = {
    "deepseek/deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "deepseek/deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct-fast",
    "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407-fast",
  };

  protected _convertModelName(model: string): string {
    return Nebius.MODEL_IDS[model] || model;
  }

  supportsCompletions(): boolean {
    return false;
  }
}

export default Nebius;
