# ToolScope × LangChain / LangGraph Examples

This directory contains examples showing how to integrate **ToolScope** into LangChain and LangGraph workflows **without changing tool semantics**.

The key idea in all examples is the same:

> **The model still sees normal LangChain tools — just fewer of them per turn.**

ToolScope runs *before* each model call and dynamically selects the most relevant subset of tools.

---

## Prerequisites

From the **ToolScope repo root**:

```bash
pip install -e .
```

For the **full agent loop** examples (which call a real model), install LangChain/LangGraph + an LLM provider integration:

```bash
pip install -U langchain langgraph langchain-openai
export OPENAI_API_KEY=...
```

---

## Examples Overview

### 0. `00_langchain_minimal.py`

**What it shows**
- The simplest possible integration: create a `ToolSelector`, filter a LangChain-style tool list, and get back the **same tool objects** (just fewer).
- Uses a dummy embedder so it can run without any external services.

**Run**
```bash
python 00_langchain_minimal.py
```

---

### 1. `01_langchain_create_agent_dynamic_tools.py`

**What it shows**
- Integration with `langchain.agents.create_agent`
- ToolScope implemented as **model-call middleware**
- Dynamic tool selection via `request.override(tools=...)`

**Why this example matters**
- Minimal code changes for existing LangChain agents
- Cleanest "drop-in" experience
- Fully preserves LangChain agent semantics

**Run**
```bash
python 01_langchain_create_agent_dynamic_tools.py
```

---

### 2. `02_langgraph_tool_provider.py`

**What it shows**
- A LangGraph-friendly *pattern*: build a `tool_provider(state) -> tools` callable.
- Useful when your LangGraph node needs a function that returns the tool list for the current turn/state.
- Still uses a dummy embedder; no model calls.

**Run**
```bash
python 02_langgraph_tool_provider.py
```

---

### 3. `03_langgraph_full_agent_loop_toolscope.py`

**What it shows**
- A complete LangGraph agent loop
- Explicit graph nodes:
  - tool selection (ToolScope)
  - model call
  - tool execution
- ToolScope runs **once per turn**, before the model sees tools

**Why this example matters**
- Most transparent and debuggable setup
- Framework-agnostic pattern (easy to adapt)
- Works even if LangChain middleware APIs change

**Run**
```bash
python 03_langgraph_full_agent_loop_toolscope.py
```

---

## Where ToolScope Fits in the Stack

```
User Prompt
   ↓
LangChain / LangGraph Agent
   ↓
ToolScope (select relevant tools for this turn)
   ↓
LLM call with filtered tools
   ↓
Tool execution
   ↓
Next turn (repeat)
```

ToolScope never:
- adds meta-tools
- wraps or proxies tool calls
- changes tool schemas

It only decides **which tools are visible to the model**.

---

## Safety & Policies

The examples demonstrate:
- `deny_tags=["dangerous"]` for hard safety filters
- tag-based allow/deny without modifying LangChain internals

You can extend this with:
- name-based policies
- environment-based policies
- per-session policies

---

## Notes

- Some examples use a tiny dummy embedder to avoid external dependencies.
  In production, replace it with:
  ```python
  toolscope.EmbeddingConfig(provider="http", ...)
  ```
- Sticky multi-turn toolsets, reranking, and observability are supported, but not all are shown in these minimal files.
