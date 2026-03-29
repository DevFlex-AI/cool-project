import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * NVIDIA NIM — optimized inference for NVIDIA GPUs
 * Models: llama-3.1-nemotron-70b, mistral-nemo-12b, etc.
 */
class Nvidia extends OpenAI {
  static providerName: ModelProvider = "nvidia" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://integrate.api.nvidia.com/v1/",
    model: "nvidia/llama-3.1-nemotron-70b-instruct",
  };

  supportsCompletions(): boolean {
    return false;
  }
}

export default Nvidia;
