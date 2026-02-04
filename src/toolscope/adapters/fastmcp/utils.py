from typing import Any, List


def extract_tools(list_tools_result: Any) -> List[Any]:
    """
    FastMCP client implementations vary. Common shapes:
      - list[tool]
      - object with `.tools`
      - dict {"tools": [...]}

    Return list[tool] in all cases.
    """
    if list_tools_result is None:
        return []
    if isinstance(list_tools_result, list):
        return list_tools_result
    if isinstance(list_tools_result, dict) and isinstance(list_tools_result.get("tools"), list):
        return list_tools_result["tools"]
    tools = getattr(list_tools_result, "tools", None)
    if isinstance(tools, list):
        return tools
    # last-resort: try to iterate
    try:
        return list(list_tools_result)
    except Exception as e:
        raise TypeError(f"Unsupported list_tools() result shape: {type(list_tools_result)!r}") from e
