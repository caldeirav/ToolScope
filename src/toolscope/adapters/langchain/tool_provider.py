from typing import Any, Callable, Optional, Sequence

from .selector import ToolSelector


def tool_provider(
    selector: ToolSelector,
    tools: Sequence[Any],
    *,
    k: int = 12,
    allow_tags: Optional[Sequence[str]] = None,
    deny_tags: Optional[Sequence[str]] = None,
    session_id_getter: Optional[Callable[[Any], Optional[str]]] = None,
    messages_getter: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Any], Sequence[Any]]:
    """
    Returns a callable(state) -> filtered_tools for LangGraph-style usage.

    You provide:
      - selector: ToolSelector (holds ToolScope index/backend/embedder)
      - tools: full tool list
      - messages_getter: extracts "messages" from state (default: state["messages"])
      - session_id_getter: extracts session id from state (default: state.get("session_id"))

    Use inside your graph nodes where you need tools for the current turn.
    """
    if messages_getter is None:
        def messages_getter(state: Any) -> Any:
            return state["messages"]

    if session_id_getter is None:
        def session_id_getter(state: Any) -> Optional[str]:
            if isinstance(state, dict):
                return state.get("session_id")
            return None

    def _provider(state: Any) -> Sequence[Any]:
        messages = messages_getter(state)
        sid = session_id_getter(state)
        return selector.select(
            messages,
            tools,
            k=k,
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            session_id=sid,
        )

    return _provider
