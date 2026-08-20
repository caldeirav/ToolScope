# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, all dev deps)
pip install -e ".[dev,st,http,milvus,fastmcp]"

# Run all unit tests
pytest

# Run a single test file
pytest tests/test_public_api.py

# Run a specific test
pytest tests/test_public_api.py::test_filter_basic -v

# Run only unit tests (skip integration tests requiring optional deps)
pytest -m "not integration"

# Run integration tests only
pytest -m integration

# Build distribution
python -m build
```

## Architecture

ToolScope is a **Tool RAG** library. It indexes tool definitions in a vector store and, at inference time, retrieves only the most relevant tools for a given prompt. The model never sees ToolScope — it just receives a smaller, better-matched tool list.

### Data flow

```
tools (OpenAI / MCP dicts or objects)
  ↓  normalize (ToolNormalizer)
CanonicalTool (name, description, input_schema, fingerprint, tags, payload)
  ↓  embed (EmbeddingProvider)
float vectors  →  ToolIndexBackend (MemoryBackend / MilvusLiteBackend)
                                      ↑
query (messages str / list)  →  embed  →  similarity search
                                      ↓
                           apply tag filters (allow/deny)
                                      ↓
                      optional reranking (cross-encoder)
                                      ↓
                   session stickiness (StickySessionConfig)
                                      ↓
                         CanonicalTool.payload (original objects)  →  caller
```

### Core layer (`src/toolscope/core/`)

| File | Responsibility |
|---|---|
| `types.py` | Protocols: `EmbeddingProvider`, `ToolNormalizer`, `ToolIndexBackend`; dataclasses `CanonicalTool`, `ToolFingerprint` |
| `index.py` | `ToolIndex` (the main stateful object) and `make_index()` factory |
| `filter.py` | Stateless convenience functions `filter_tools()` / `filter_tools_with_trace()` — each call creates a fresh `ToolIndex` |
| `normalize.py` | `AutoToolNormalizer` (auto-detects schema), `McpToolNormalizer`, `OpenAIToolDictNormalizer` |
| `fingerprint.py` | SHA-256 fingerprint of (name, description, schema) — used as vector primary key |
| `embeddings.py` | Resolves an `EmbeddingProvider` from `EmbeddingConfig`; raises `EmbeddingNotConfiguredError` when none is configured |
| `embedding_config.py` | `EmbeddingConfig` dataclass (provider, model, endpoint, …) |
| `text.py` | `ToolTextConfig` — controls what text is embedded (name, description, schema, truncation, preprocessors) |
| `session.py` | `StickySessionConfig` / `SessionCache` — reuse tool selections across multi-turn conversations |
| `rerankers.py` / `reranking_config.py` | Optional cross-encoder reranking after vector retrieval |
| `observability.py` | `ToolScopeTrace` — per-call timing and decision data; `TraceSink` for streaming traces out |
| `cache.py` | `EmbeddingCache` — fingerprint → vector, avoids re-embedding unchanged tools |
| `backends/memory.py` | `MemoryBackend` — default in-process vector store |
| `backends/milvus_lite.py` | `MilvusLiteBackend` — local persistent vector store (optional dep `pymilvus`) |

### Public API (`src/toolscope/__init__.py`)

Three top-level entry points:

- `toolscope.filter(messages, tools, *, embedder=..., k=12, ...)` — one-shot stateless filter
- `toolscope.filter_with_trace(...)` — same, returns `(tools, ToolScopeTrace)`
- `toolscope.index(tools, ...)` → `ToolIndex` — stateful; call `.filter()` / `.filter_with_trace()` per turn

### Adapters (`src/toolscope/adapters/`)

**FastMCP** (`adapters/fastmcp/`): `ToolScopeFastMCPClient` wraps any FastMCP client. Exposes `await wrapper.list_tools(messages, k=...)` and `await wrapper.call_tool(name, args)`. Auto-refreshes the index when tools change (via `mark_dirty()` or fingerprint diff on `auto_refresh=True`).

**LangChain/LangGraph** (`adapters/langchain/`): `ToolSelector` wraps a `ToolIndex` with LangChain-aware tool handling. `make_toolscope_tool_selection_middleware()` returns a LangChain agent middleware that overrides the tool list on each model call transparently.

### Tool normalization

`AutoToolNormalizer` detects the schema per tool:
- OpenAI dict: `{"type": "function", "function": {"name", "description", "parameters"}}`
- MCP dict: `{"name", "description", "inputSchema"}`
- MCP object: anything with `.name` and `.inputSchema` attributes

Tags are read from `toolscope_tags`, `tags`, or `annotations.tags` keys/attrs (non-breaking extension point).

### Testing

Tests use `DummyEmbedder` (deterministic 3D vector from string features) and `DummyReranker` from `tests/conftest.py`. Integration tests (marked `@pytest.mark.integration`) require optional deps like `fastmcp` or `pymilvus`.
