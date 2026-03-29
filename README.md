# PearAI — Full Codebase

AI-powered code editor built on VS Code. This monorepo contains all PearAI components:

| Directory | Description |
|-----------|-------------|
| `pearai-app/` | VS Code fork — the main editor application |
| `pearai-submodule/` | AI extension (continue.dev fork) — LLM providers, chat, autocomplete |
| `PearAI-Roo-Code/` | Roo Code AI agent integration |
| `pearai-server/` | Backend server |
| `pearai-mcp/` | MCP (Model Context Protocol) integration |

## Updated AI Providers (pearai-submodule)

- **Anthropic** — Claude 3.7 Sonnet, extended thinking, claude-3-5-haiku
- **Gemini** — 2.0 Flash, 2.5 Pro/Flash, 2M context, native thinking
- **OpenAI** — o-series (o1/o3/o4-mini), reasoning_effort, gpt-4.5
- **DeepSeek** — V3 (deepseek-chat), R1 (deepseek-reasoner)
- **xAI** — Grok-2, Grok-2-mini *(new)*
- **Cerebras** — Ultra-fast inference *(new)*
- **Nebius** — GPU cloud *(new)*
- **SambaNova** — High-throughput inference *(new)*
- **NVIDIA NIM** *(new)*
- **SiliconFlow** *(new)*
- **Venice AI** — Privacy-focused *(new)*
- **vLLM** — Self-hosted *(new)*

## Setup

```bash
git clone https://github.com/DevFlex-AI/cool-project
cd cool-project
bash setup-app-dev.sh
```
