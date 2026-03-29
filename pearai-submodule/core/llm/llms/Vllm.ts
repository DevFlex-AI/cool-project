import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * vLLM — self-hosted high-throughput LLM serving
 * Run any HuggingFace model locally with OpenAI-compatible API
 */
class Vllm extends OpenAI {
  static providerName: ModelProvider = "vllm" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "http://localhost:8000/v1/",
    model: "meta-llama/Llama-3.1-8B-Instruct",
  };
}

export default Vllm;
