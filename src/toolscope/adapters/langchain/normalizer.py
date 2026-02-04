from typing import Any, Dict, List, Sequence, Tuple

from ...core.types import CanonicalTool, ToolNormalizer
from ...core.fingerprint import fingerprint_tool


def _maybe_schema_from_args_schema(args_schema: Any) -> Dict[str, Any]:
    """
    Supports:
      - Pydantic v2: model_json_schema()
      - Pydantic v1: schema()
    Returns {} if unknown.
    """
    if args_schema is None:
        return {}
    fn = getattr(args_schema, "model_json_schema", None)
    if callable(fn):
        try:
            s = fn()
            return s if isinstance(s, dict) else {}
        except Exception:
            return {}
    fn = getattr(args_schema, "schema", None)
    if callable(fn):
        try:
            s = fn()
            return s if isinstance(s, dict) else {}
        except Exception:
            return {}
    return {}


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def _tool_desc(tool: Any) -> str:
    # LangChain tools typically have `.description`
    return str(getattr(tool, "description", "") or "")


def _tool_tags(tool: Any) -> Tuple[str, ...]:
    tags = getattr(tool, "tags", None)
    if isinstance(tags, (list, tuple)):
        out: List[str] = []
        for t in tags:
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
        return tuple(out)
    return ()


def _tool_schema(tool: Any) -> Dict[str, Any]:
    # Common in langchain: args_schema (pydantic model class)
    args_schema = getattr(tool, "args_schema", None)
    schema = _maybe_schema_from_args_schema(args_schema)
    if schema:
        return schema

    # Some tools expose `.args` or `.parameters`
    for key in ("parameters", "args", "input_schema", "inputSchema"):
        v = getattr(tool, key, None)
        if isinstance(v, dict):
            return v

    # Some expose get_input_schema()
    fn = getattr(tool, "get_input_schema", None)
    if callable(fn):
        try:
            s = fn()
            if isinstance(s, dict):
                return s
        except Exception:
            pass

    return {}


class LangChainToolNormalizer(ToolNormalizer):
    """
    Normalizes LangChain BaseTool / StructuredTool objects (and duck-typed equivalents).

    Requirements (duck-typed):
      - tool.name: str
      - tool.description: str
    Optional:
      - tool.args_schema: pydantic model class
      - tool.tags: list[str]
    """

    def normalize(self, tools: Sequence[Any]) -> List[CanonicalTool]:
        out: List[CanonicalTool] = []
        for t in tools:
            name = _tool_name(t)
            desc = _tool_desc(t)
            schema = _tool_schema(t)
            tags = _tool_tags(t)

            fp = fingerprint_tool(name, desc, schema)

            out.append(
                CanonicalTool(
                    name=name,
                    description=desc,
                    input_schema=schema,
                    tags=tags,
                    fingerprint=fp,
                    payload=t,
                )
            )
        return out

    def denormalize(self, canonical_tools: Sequence[CanonicalTool]) -> List[Any]:
        # We return the original tool objects.
        out: List[Any] = []
        for ct in canonical_tools:
            if ct.payload is None:
                raise ValueError("CanonicalTool.payload missing; cannot round-trip tool objects.")
            out.append(ct.payload)
        return out
