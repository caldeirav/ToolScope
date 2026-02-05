# ToolScope

[![PyPI](https://img.shields.io/pypi/v/toolscope)](https://pypi.org/project/toolscope/)
[![Python](https://img.shields.io/pypi/pyversions/toolscope)](https://pypi.org/project/toolscope/)
[![License](https://img.shields.io/pypi/l/toolscope)](LICENSE)

**ToolScope** solves a core scalability problem in tool-using agents:

> As the number of tools grows, LLMs become worse at selecting the right one.

In addition to degraded accuracy and reliability, including a large number of tools in the prompt consumes context budget and can lead to prompt bloat.
This is especially evident with small models.

ToolScope addresses this problem by filtering tools per prompt using semantic retrieval - the same way RAG does for text.

*ToolScope does not change how the model interacts with tools, introduces no meta-tools and no framework lock-in.*

---

## Who should use ToolScope?

ToolScope is for you if:
- you have >20 tools, coming from MCP servers or registries
- you want to keep using standard agent frameworks
- you want predictable, debuggable behavior
- you don’t want meta-tools

---

## Quickstart

### 1. Install

```bash
pip install toolscope
```

(or from source)

```bash
pip install -e .
```


### 2. Minimal filtering

```python
import toolscope

class TinyEmbedder:
    def embed_texts(self, texts):
        return [[len(t) % 97, t.count("jira"), sum(map(ord, t)) % 101] for t in texts]

tools = [
    {"name": "jira_create_issue", "description": "Create a Jira issue", "inputSchema": {}},
    {"name": "confluence_search", "description": "Search Confluence pages", "inputSchema": {}},
]

filtered = toolscope.filter(
    messages=[{"role": "user", "content": "Create a Jira ticket"}],
    tools=tools,
    embedder=TinyEmbedder(),
    k=1,
)

print(filtered)  # same tools, fewer of them
```

or, with an embedding configuration (requires `sentence-transformers`):
```python
import toolscope

embedding_config = toolscope.EmbeddingConfig(
    provider="sentence-transformers",
    model="sentence-transformers/all-MiniLM-L6-v2",
    allow_download=False,
)

tools = [
    {"name": "jira_create_issue", "description": "Create a Jira issue", "inputSchema": {}},
    {"name": "confluence_search", "description": "Search Confluence pages", "inputSchema": {}},
]

filtered = toolscope.filter(
    messages=[{"role": "user", "content": "Create a Jira ticket"}],
    tools=tools,
    embedding=embedding_config,
    k=1,
)

print(filtered)  # same tools, fewer of them
```


### 3. Indexed (stateful) usage

```python
idx = toolscope.index(
    tools,
    embedder=TinyEmbedder(),
)

filtered = idx.filter(
    messages="Create a Jira ticket",
    k=1,
)
```

Also check out the usage examples for [LangChain](./examples/langchain/01_langchain_create_agent_dynamic_tools.py), [LangGraph](./examples/langchain/03_langgraph_full_agent_loop_toolscope.py) and [FastMCP](./examples/fastmcp/00_fastmcp_minimal.py).

---

## Core Concepts and Features

### Canonical tools
ToolScope normalizes tools from many schemas into a canonical form:
- name
- description
- input schema
- tags
- fingerprint

Original tool objects are preserved and returned unchanged.

---

### Embeddings

ToolScope supports **pluggable embedding backends**.

#### Option A: Provide your own embedder (recommended default)

```python
class MyEmbedder:
    def embed_texts(self, texts): ...
```

#### Option B: Use `EmbeddingConfig` (HTTP, OpenAI-style, etc.)

```python
toolscope.EmbeddingConfig(
    provider="http",
    endpoint="http://localhost:8000/embed",
    model="my-embedding-model",
)
```

ToolScope **never downloads models behind your back**.

---

### Tool text control

You control what text is embedded:

```python
toolscope.ToolTextConfig(
    use_name=True,
    use_description=True,
    use_schema=False,
    truncate=256,
)
```

**Defaults (battle-tested):**
- name + description only
- truncate to 256 chars
- no preprocessing

---

## Advanced Features

### ✅ Allow / deny filters

```python
idx.filter(
    messages,
    allow_tags=["jira"],
    deny_tags=["dangerous"],
)
```

---

### 🔁 Sticky toolsets (multi-turn sessions)

Reuse tools across turns when the query stays similar:

```python
toolscope.StickySessionConfig(
    enabled=True,
    similarity_threshold_reuse=0.95,
    similarity_threshold_refresh=0.8,
    sticky_keep=2,
)
```

This reduces latency and improves consistency.

---

### 🧠 Reranking

Boost retrieval quality using a cross-encoder:

```python
toolscope.RerankingConfig(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    pool_size=20,
)
```

Not enabled by default — you opt in explicitly.

---

### 📊 Observability

Inspect what ToolScope is doing:

```python
tools, trace = idx.filter_with_trace(messages)
print(trace)
```

Includes:
- candidate counts
- timings
- allow/deny decisions
- reranking effects

---

## Backends

### In-memory (default)

Fast, simple, zero dependencies.

```python
toolscope.MemoryBackend()
```

---

### Milvus Lite

Persistent, scalable local vector DB:

```python
toolscope.MilvusLiteBackend(path="./toolscope.db")
```

ToolScope is backend-agnostic; more vector DBs can be added.

---

## Adapters (Plug & Play)

ToolScope integrates cleanly with popular agent stacks.

### LangChain / LangGraph

- full agent loops
- per-turn tool filtering
- middleware-based integration

```python
from toolscope.adapters.langchain import (
    ToolSelector,
    make_toolscope_tool_selection_middleware,
)
```

See:
```
examples/langchain/
```

---

### FastMCP

- drop-in MCP client wrapper
- supports multi-server clients
- reacts to tools/list_changed notifications

```python
from toolscope.adapters.fastmcp import ToolScopeFastMCPClient
```

See:
```
examples/fastmcp/
```

---

## Status & roadmap

ToolScope is actively developed.

Adapters:
- ✅ LangGraph
- ✅ FastMCP
- ⏳ Llama Stack
- ⏳ LlamaIndex
- ⏳ AutoGen
- ⏳ CrewAI
- ⏳ Haystack

Other features:
- additional vector DB backends
- more MCP normalizers
- richer observability sinks

---

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
