"""
Convert BFCL / OpenAI tool dicts to LangChain StructuredTools and back.

Dummy ``func`` implementations are never executed — BFCL Multiple is AST-only.
"""

from __future__ import annotations

import json
import keyword
import re
from typing import Any, Dict, List, Optional, Sequence

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "float": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "dict": dict,
}

_SAFE_NAME = re.compile(r"[^0-9A-Za-z_-]+")


def tool_name(tool: Any) -> str:
    """Name of an OpenAI-format dict or a LangChain-style tool object."""
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        return str(tool.get("name") or "")
    md = getattr(tool, "metadata", None) or {}
    if isinstance(md, dict) and md.get("original_name"):
        return str(md["original_name"])
    return str(getattr(tool, "name", "") or "")


def original_tool_name(tool: Any) -> str:
    return tool_name(tool)


def safe_tool_name(name: str) -> str:
    """OpenAI / Gemini tool names: ``^[a-zA-Z0-9_-]{1,64}$``."""
    cleaned = _SAFE_NAME.sub("_", name).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = "fn_" + cleaned
    return (cleaned or "tool")[:64]


def tools_to_openai(tools: Sequence[Any]) -> List[Dict]:
    """Round-trip LangChain tools (or mixed lists) back to OpenAI dicts."""
    out: List[Dict] = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function":
            out.append(t)
            continue
        md = getattr(t, "metadata", None) or {}
        if isinstance(md, dict) and isinstance(md.get("openai_tool"), dict):
            out.append(md["openai_tool"])
            continue
        name = tool_name(t)
        desc = str(getattr(t, "description", "") or "")
        params = getattr(t, "args", None)
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params,
            },
        })
    return out


def _pydantic_ident(pname: str, used: set[str]) -> str:
    """Pydantic v2 field names cannot be keywords or start with '_'."""
    ident = re.sub(r"[^0-9A-Za-z_]", "_", pname)
    ident = ident.strip("_") or "param"
    if ident[0].isdigit():
        ident = "p_" + ident
    if keyword.iskeyword(ident):
        ident = ident + "_"
    base = ident
    n = 2
    while ident in used:
        ident = f"{base}_{n}"
        n += 1
    used.add(ident)
    return ident


def openai_tool_to_structured(tool: Dict) -> Any:
    """Build a LangChain StructuredTool from an OpenAI-format function dict."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict, Field, create_model

    fn = tool.get("function") or {}
    original = str(fn.get("name") or "tool")
    name = safe_tool_name(original)
    desc = str(fn.get("description") or original)
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])

    fields: Dict[str, Any] = {}
    used_idents: set[str] = set()
    for pname, pdef in properties.items():
        if not isinstance(pname, str) or not pname:
            continue
        pdef = pdef if isinstance(pdef, dict) else {}
        py_type = _JSON_TO_PY.get(str(pdef.get("type", "string")).lower(), str)
        pdesc = str(pdef.get("description") or "")
        ident = _pydantic_ident(pname, used_idents)
        if pname in required:
            fields[ident] = (py_type, Field(..., description=pdesc, alias=pname))
        else:
            fields[ident] = (Optional[py_type], Field(None, description=pdesc, alias=pname))

    class _ToolArgs(BaseModel):
        model_config = ConfigDict(extra="allow", populate_by_name=True)

    ArgsSchema = create_model(f"{name}Input", __base__=_ToolArgs, **fields)

    def _noop(**kwargs: Any) -> str:
        return json.dumps(kwargs)

    _noop.__name__ = name
    _noop.__doc__ = desc

    st = StructuredTool.from_function(
        func=_noop,
        name=name,
        description=desc,
        args_schema=ArgsSchema,
    )
    st.metadata = {"openai_tool": tool, "original_name": original}
    return st


def catalog_to_langchain(openai_tools: Sequence[Dict]) -> List[Any]:
    out: List[Any] = []
    for t in openai_tools:
        try:
            out.append(openai_tool_to_structured(t))
        except Exception as exc:
            raise RuntimeError(
                f"Failed converting tool {tool_name(t)!r} to StructuredTool: {exc}"
            ) from exc
    return out


def dedupe_lc_tools_by_safe_name(tools: Sequence[Any]) -> List[Any]:
    """Keep the first tool per Gemini/OpenAI-safe name (``^[A-Za-z0-9_-]{1,64}$``).

    Retrieval still sees original names (``car.rental`` vs ``car_rental``).
    ``bind_tools`` cannot: both sanitize to ``car_rental``. First-seen wins;
    ``metadata.original_name`` on the kept tool is unchanged.
    """
    seen: set[str] = set()
    out: List[Any] = []
    for t in tools:
        bound_name = str(getattr(t, "name", "") or "") or safe_tool_name(tool_name(t))
        if bound_name in seen:
            continue
        seen.add(bound_name)
        out.append(t)
    return out
