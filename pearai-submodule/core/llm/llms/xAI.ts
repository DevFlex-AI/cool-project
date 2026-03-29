import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

/**
 * xAI (Grok) provider — OpenAI-compatible API
 * Supports: grok-2, grok-2-mini, grok-beta, grok-vision-beta
 */
class xAI extends OpenAI {
  static providerName: ModelProvider = "xAI" as ModelProvider;
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.x.ai/v1/",
    model: "grok-2",
  };

  supportsCompletions(): boolean {
    return false;
  }
}

export default xAI;
