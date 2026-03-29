import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * SiliconFlow — Chinese AI cloud with top open-source models
 * Models: deepseek-v3, qwen2.5-72b, llama3.1-70b, etc.
 */
class SiliconFlow extends OpenAI {
  static providerName: ModelProvider = "siliconflow" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.siliconflow.cn/v1/",
    model: "deepseek-ai/DeepSeek-V3",
  };

  supportsCompletions(): boolean {
    return false;
  }
}

export default SiliconFlow;
