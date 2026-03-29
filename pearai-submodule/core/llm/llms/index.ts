import Handlebars from "handlebars";
import { v4 as uuidv4 } from "uuid";
import {
  BaseCompletionOptions,
  IdeSettings,
  ILLM,
  LLMOptions,
  ModelDescription,
  PearAuth,
} from "../../index.js";
import { DEFAULT_MAX_TOKENS } from "../constants.js";
import { BaseLLM } from "../index.js";

// Core providers (original)
import Anthropic from "./Anthropic.js";
import Azure from "./Azure.js";
import Bedrock from "./Bedrock.js";
import Cloudflare from "./Cloudflare.js";
import Cohere from "./Cohere.js";
import DeepInfra from "./DeepInfra.js";
import Deepseek from "./Deepseek.js";
import Fireworks from "./Fireworks.js";
import Flowise from "./Flowise.js";
import FreeTrial from "./FreeTrial.js";
import Gemini from "./Gemini.js";
import Groq from "./Groq.js";
import HuggingFaceInferenceAPI from "./HuggingFaceInferenceAPI.js";
import HuggingFaceTGI from "./HuggingFaceTGI.js";
import LMStudio from "./LMStudio.js";
import LlamaCpp from "./LlamaCpp.js";
import Llamafile from "./Llamafile.js";
import Mistral from "./Mistral.js";
import Msty from "./Msty.js";
import Ollama from "./Ollama.js";
import OpenAI from "./OpenAI.js";
import OpenRouter from "./OpenRouter.js";
import Replicate from "./Replicate.js";
import TextGenWebUI from "./TextGenWebUI.js";
import Together from "./Together.js";
import WatsonX from "./WatsonX.js";
import ContinueProxy from "./stubs/ContinueProxy.js";
import PearAIServer from "./PearAIServer.js";

// New providers (2024-2025)
import Cerebras from "./Cerebras.js";
import Nebius from "./Nebius.js";
import Novita from "./Novita.js";
import Nvidia from "./Nvidia.js";
import SambaNova from "./SambaNova.js";
import SiliconFlow from "./SiliconFlow.js";
import Venice from "./Venice.js";
import Vllm from "./Vllm.js";
import xAI from "./xAI.js";


function convertToLetter(num: number): string {
  let result = "";
  while (num > 0) {
    const remainder = (num - 1) % 26;
    result = String.fromCharCode(97 + remainder) + result;
    num = Math.floor((num - 1) / 26);
  }
  return result;
}

const getHandlebarsVars = (
  value: string,
): [string, { [key: string]: string }] => {
  const ast = Handlebars.parse(value);

  const keysToFilepath: { [key: string]: string } = {};
  let keyIndex = 1;
  for (const i in ast.body) {
    if (ast.body[i].type === "MustacheStatement") {
      const letter = convertToLetter(keyIndex);
      keysToFilepath[letter] = (ast.body[i] as any).path.original;
      value = value.replace(
        new RegExp(`{{\\s*${(ast.body[i] as any).path.original}\\s*}}`),
        `{{${letter}}}`,
      );
      keyIndex++;
    }
  }
  return [value, keysToFilepath];
};

export async function renderTemplatedString(
  template: string,
  readFile: (filepath: string) => Promise<string>,
  inputData: any,
  helpers?: [string, Handlebars.HelperDelegate][],
): Promise<string> {
  const promises: { [key: string]: Promise<string> } = {};
  if (helpers) {
    for (const [name, helper] of helpers) {
      Handlebars.registerHelper(name, (...args) => {
        const id = uuidv4();
        promises[id] = helper(...args);
        return `__${id}__`;
      });
    }
  }

  const [newTemplate, vars] = getHandlebarsVars(template);
  const data: any = { ...inputData };
  for (const key in vars) {
    const fileContents = await readFile(vars[key]);
    data[key] = fileContents || (inputData[vars[key]] ?? vars[key]);
  }
  const templateFn = Handlebars.compile(newTemplate);
  let final = templateFn(data);

  await Promise.all(Object.values(promises));
  for (const id in promises) {
    final = final.replace(`__${id}__`, await promises[id]);
  }

  return final;
}

// All available LLM providers — ordered: PearAI first, then alphabetical
export const LLMs = [
  // PearAI native
  PearAIServer,
  FreeTrial,
  // OpenAI family
  OpenAI,
  Azure,
  // Anthropic
  Anthropic,
  // Google
  Gemini,
  // Open-source / hosted inference
  Ollama,
  LMStudio,
  Llamafile,
  LlamaCpp,
  Msty,
  Vllm,
  // Hosted API providers (alphabetical)
  Bedrock,
  Cerebras,
  Cloudflare,
  Cohere,
  DeepInfra,
  Deepseek,
  Fireworks,
  Flowise,
  Groq,
  HuggingFaceInferenceAPI,
  HuggingFaceTGI,
  Mistral,
  Nebius,
  Novita,
  Nvidia,
  OpenRouter,
  Replicate,
  SambaNova,
  SiliconFlow,
  TextGenWebUI,
  Together,
  Venice,
  WatsonX,
  xAI,
  // Infrastructure
  ContinueProxy,
];

export async function llmFromDescription(
  desc: ModelDescription,
  readFile: (filepath: string) => Promise<string>,
  uniqueId: string,
  ideSettings: IdeSettings,
  writeLog: (log: string) => Promise<void>,
  completionOptions?: BaseCompletionOptions,
  systemMessage?: string,
  getCurrentDirectory?: () => Promise<string>,
  getCredentials?: () => Promise<PearAuth | undefined>,
  setCredentials?: (auth: PearAuth) => Promise<void>,
): Promise<BaseLLM | undefined> {
  const cls = LLMs.find((llm) => llm.providerName === desc.provider);

  if (!cls) {
    console.warn(`[PearAI] Unknown LLM provider: "${desc.provider}". Available: ${LLMs.map(l => l.providerName).join(", ")}`);
    return undefined;
  }

  const finalCompletionOptions = {
    ...completionOptions,
    ...desc.completionOptions,
  };

  systemMessage = desc.systemMessage ?? systemMessage;
  if (systemMessage !== undefined) {
    systemMessage = await renderTemplatedString(systemMessage, readFile, {});
  }

  let options: LLMOptions = {
    ...desc,
    completionOptions: {
      ...finalCompletionOptions,
      model: (desc.model || cls.defaultOptions?.model) ?? "codellama-7b",
      maxTokens:
        finalCompletionOptions.maxTokens ??
        cls.defaultOptions?.completionOptions?.maxTokens ??
        DEFAULT_MAX_TOKENS,
    },
    systemMessage,
    writeLog,
    uniqueId,
    getCurrentDirectory,
    getCredentials,
    setCredentials,
  };

  if (desc.provider === "continue-proxy") {
    options.apiKey = ideSettings.userToken;
    if (ideSettings.remoteConfigServerUrl) {
      options.apiBase = new URL(
        "/proxy/v1",
        ideSettings.remoteConfigServerUrl,
      ).toString();
    }
  }

  return new cls(options);
}

export function llmFromProviderAndOptions(
  providerName: string,
  llmOptions: LLMOptions,
): ILLM {
  const cls = LLMs.find((llm) => llm.providerName === providerName);
  if (!cls) {
    throw new Error(`[PearAI] Unknown LLM provider: "${providerName}"`);
  }
  return new cls(llmOptions);
}
