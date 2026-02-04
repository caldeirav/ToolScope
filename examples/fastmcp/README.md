# ToolScope × FastMCP Examples

This folder contains examples showing how to integrate **ToolScope** with **FastMCP** clients **without changing tool semantics**.

ToolScope only filters the tool list returned from `tools/list` (per turn). The model still sees normal MCP tools — just fewer of them.

---

## Prerequisites

From the **ToolScope repo root**:

```bash
pip install -e .
```

To run examples against a real FastMCP setup:

```bash
pip install -U fastmcp
```

---

## Examples

### 0. `00_fastmcp_minimal.py`

**What it shows**
- Wrap an existing `fastmcp.Client` (or any FastMCP-like client with `list_tools` and `call_tool`)
- Filter tools per prompt (`messages`) using ToolScope
- Call tools via the wrapper (passthrough)

**Run**
```bash
python 00_fastmcp_minimal.py
```

---

### 1. `01_fastmcp_multi_server.py`

**What it shows**
- A multi-server FastMCP client setup (tool names are typically prefixed by server)
- ToolScope filtering still works the same (it is transport-agnostic)

**Run**
```bash
python 01_fastmcp_multi_server.py
```

---

### 2. `02_fastmcp_tools_list_changed.py`

**What it shows**
- How to hook `notifications/tools/list_changed` (when available) so ToolScope refreshes its index automatically
- Includes a safe fallback if your FastMCP client does not allow setting a message handler after construction

**Run**
```bash
python 02_fastmcp_tools_list_changed.py
```

---

## Where ToolScope Fits

```
Your agent/framework
   ↓ (asks for tools/list)
ToolScope FastMCP wrapper
   ↓ (calls upstream list_tools)
FastMCP client
   ↓
MCP servers
```

ToolScope never:
- adds meta-tools
- changes tool schemas
- proxies tool calls

It only chooses **which tools are visible to the model**.

---

## Notes

- These examples use a tiny dummy embedder so they can run without external services.
  In production, use:
  ```python
  toolscope.EmbeddingConfig(provider="http", endpoint="http://.../embed", model="...")
  ```
- Reranking, allow/deny tag filters, and sticky sessions are supported by the wrapper as optional arguments.
