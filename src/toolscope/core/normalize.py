from typing import Any, List, Sequence, Tuple, Dict

from .types import CanonicalTool, ToolNormalizer
from .fingerprint import fingerprint_tool


def _get_attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _extract_tags_from_mapping(m: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Best-effort tags extraction. We intentionally keep this simple and non-invasive.

    Supported keys (in order):
      - "toolscope_tags" (preferred)
      - "tags"
      - "annotations": {"tags": [...]}
    """
    tags = None
    if isinstance(m.get("toolscope_tags"), (list, tuple)):
        tags = m.get("toolscope_tags")
    elif isinstance(m.get("tags"), (list, tuple)):
        tags = m.get("tags")
    else:
        ann = m.get("annotations")
        if isinstance(ann, dict) and isinstance(ann.get("tags"), (list, tuple)):
            tags = ann.get("tags")

    if not tags:
        return ()

    out: list[str] = []
    for t in tags:
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return tuple(out)


def _is_openai_tool_dict(t: Any) -> bool:
    # Expected shape:
    # {"type":"function","function":{"name":..,"description":..,"parameters":..}}
    return (
        isinstance(t, dict)
        and t.get("type") == "function"
        and isinstance(t.get("function"), dict)
        and isinstance((t.get("function") or {}).get("name"), (str, type(None)))
        and "parameters" in (t.get("function") or {})
    )


def _is_mcp_tool_dict(t: Any) -> bool:
    # Expected shape (canonical MCP Tool):
    # {"name":..,"description":..,"inputSchema":{...}, ...}
    return (
        isinstance(t, dict)
        and isinstance(t.get("name"), str)
        and "inputSchema" in t
        and isinstance(t.get("inputSchema"), dict)
    )


def _is_mcp_tool_object(t: Any) -> bool:
    # Common object form: has attributes name, description, inputSchema
    return (
        _get_attr(t, "name") is not None
        and _get_attr(t, "inputSchema") is not None
    )


class OpenAIToolDictNormalizer(ToolNormalizer):
    """
    Normalizer for OpenAI function tool dicts:
      {"type":"function","function":{"name":..., "description":..., "parameters":{...}}}
    """
    def normalize(self, tools: Sequence[Any]) -> List[CanonicalTool]:
        out: List[CanonicalTool] = []
        for t in tools:
            if not _is_openai_tool_dict(t):
                raise TypeError("OpenAIToolDictNormalizer expects OpenAI-like tool dicts.")

            fn = t.get("function") or {}
            name = str(fn.get("name") or "")
            desc = str(fn.get("description") or "")
            params = fn.get("parameters") or {}

            tags = ()
            if isinstance(t, dict):
                tags = _extract_tags_from_mapping(t)

            fp = fingerprint_tool(name, desc, params)
            out.append(
                CanonicalTool(
                    name=name,
                    description=desc,
                    input_schema=params,
                    tags=tags,
                    fingerprint=fp,
                    payload=t,
                )
            )
        return out

    def denormalize(self, canonical_tools: Sequence[CanonicalTool]) -> List[Any]:
        out: List[Any] = []
        for ct in canonical_tools:
            if ct.payload is None:
                raise ValueError("CanonicalTool.payload missing; cannot round-trip.")
            out.append(ct.payload)
        return out


class McpToolNormalizer(ToolNormalizer):
    """
    Normalizer for MCP tool descriptors (dicts or objects).

    Dict shape:
      {"name": str, "description": str|None, "inputSchema": dict, "annotations": ...}

    Object shape:
      tool.name, tool.description, tool.inputSchema (and optionally tool.annotations)

    We preserve the original object/dict in payload for round-trip.
    """
    def normalize(self, tools: Sequence[Any]) -> List[CanonicalTool]:
        out: List[CanonicalTool] = []
        for t in tools:
            if _is_mcp_tool_dict(t):
                name = t.get("name") or ""
                desc = t.get("description") or ""
                schema = t.get("inputSchema") or {}

                # optional tags: not standard in MCP
                tags = _extract_tags_from_mapping(t)

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
                continue

            if _is_mcp_tool_object(t):
                name = _get_attr(t, "name") or ""
                desc = _get_attr(t, "description") or ""
                schema = _get_attr(t, "inputSchema") or {}

                raw_tags = _get_attr(t, "tags")
                tags = ()
                if isinstance(raw_tags, (list, tuple)):
                    tags = tuple(str(x).strip() for x in raw_tags if isinstance(x, str) and x.strip())

                fp = fingerprint_tool(str(name), str(desc), schema)
                out.append(
                    CanonicalTool(
                        name=str(name),
                        description=str(desc),
                        input_schema=schema,
                        tags=tags,
                        fingerprint=fp,
                        payload=t,
                    )
                )
                continue

            raise TypeError("McpToolNormalizer expects MCP tool dicts or objects with .name/.inputSchema.")
        return out

    def denormalize(self, canonical_tools: Sequence[CanonicalTool]) -> List[Any]:
        out: List[Any] = []
        for ct in canonical_tools:
            if ct.payload is None:
                raise ValueError("CanonicalTool.payload missing; cannot round-trip.")
            out.append(ct.payload)
        return out


class AutoToolNormalizer(ToolNormalizer):
    """
    Auto-detect tool schema per tool and normalize accordingly.

    Supports:
      - OpenAI function tool dicts
      - MCP tool dicts
      - MCP tool objects (attrs: name, inputSchema)

    Behavior:
      - Strict: if a tool doesn't match a known shape, raises TypeError.
    """
    def __init__(self) -> None:
        self._openai = OpenAIToolDictNormalizer()
        self._mcp = McpToolNormalizer()

    def normalize(self, tools: Sequence[Any]) -> List[CanonicalTool]:
        out: List[CanonicalTool] = []
        for t in tools:
            if _is_openai_tool_dict(t):
                out.extend(self._openai.normalize([t]))
            elif _is_mcp_tool_dict(t) or _is_mcp_tool_object(t):
                out.extend(self._mcp.normalize([t]))
            else:
                raise TypeError(
                    "AutoToolNormalizer could not detect tool schema. "
                    "Supported: OpenAI tool dicts, MCP tool dicts, MCP tool objects."
                )
        return out

    def denormalize(self, canonical_tools: Sequence[CanonicalTool]) -> List[Any]:
        # We always round-trip via CanonicalTool.payload, so no need to know schema here.
        out: List[Any] = []
        for ct in canonical_tools:
            if ct.payload is None:
                raise ValueError("CanonicalTool.payload missing; cannot round-trip.")
            out.append(ct.payload)
        return out
