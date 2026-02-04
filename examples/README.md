# ToolScope Examples

This folder contains runnable examples demonstrating **all major ToolScope features**, from the simplest usage to advanced production patterns.

All examples assume you are running from the **repository root** with ToolScope installed in editable mode.

---

## Setup

Create and activate a virtual environment, then install ToolScope:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -e .
```

For examples that require optional dependencies, see the sections below.

---

## Example Index

| File | What it demonstrates |
|-----|----------------------|
| `00_minimal_filter.py` | Minimal stateless `toolscope.filter(...)` |
| `01_index_and_filter.py` | Stateful `toolscope.index(...).filter(...)` |
| `02_auto_normalizer_mixed_tools.py` | Auto normalizer with mixed OpenAI + MCP tools |
| `03_tool_text_config.py` | Tool text fields, truncation, and preprocessing |
| `04_allow_deny_tags.py` | Allow / deny tag filtering |
| `05_embedding_http_provider.py` | Production-style HTTP embedding provider |
| `06_embedding_sentence_transformers.py` | Sentence-Transformers embeddings (optional) |
| `07_reranking_config.py` | Optional reranking configuration |
| `08_session_sticky_tools.py` | Session caching + sticky toolsets |
| `09_observability_trace.py` | Structured observability with traces |
| `10_end_to_end_agent_loop.py` | End-to-end multi-turn agent-style loop |

---

## Running the Examples

### Minimal example
```bash
python examples/00_minimal_filter.py
```

### Stateful index usage
```bash
python examples/01_index_and_filter.py
```

### Allow / deny tag filtering
```bash
python examples/04_allow_deny_tags.py
```

### Observability & tracing
```bash
python examples/09_observability_trace.py
```

### Full agent-style loop
```bash
python examples/10_end_to_end_agent_loop.py
```

---

## Optional: Sentence-Transformers Embeddings

Some examples require Sentence-Transformers:

```bash
pip install -e ".[st]"
```

Then run:

```bash
python examples/06_embedding_sentence_transformers.py
python examples/07_reranking_config.py
```

⚠️ By default, ToolScope **does not download models automatically**.  
You must either pre-download models or explicitly enable downloads via configuration.

---

## Optional: Milvus Lite Backend

To experiment with a lightweight vector database backend:

```bash
pip install -e ".[milvus]"
```

Then create your own script or adapt an example to use:

```python
backend=toolscope.MilvusLiteBackend(...)
```

Milvus examples are intentionally not run by default.

---

## Design Notes

- All examples intentionally avoid framework-specific agent logic.
- ToolScope never changes how tools are *presented* to the LLM — only **which subset** is shown.
- Features like reranking, session caching, and observability are **opt-in**.
- Defaults are conservative and production-safe.

---

## Recommended Reading Order

If you're new to ToolScope:

1. `00_minimal_filter.py`
2. `01_index_and_filter.py`
3. `04_allow_deny_tags.py`
4. `08_session_sticky_tools.py`
5. `09_observability_trace.py`
6. `10_end_to_end_agent_loop.py`

---

If you have questions or want to propose new examples, open an issue or start with a minimal reproduction script.
