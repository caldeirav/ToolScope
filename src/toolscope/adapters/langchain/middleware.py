from typing import Any, Callable, Optional, Sequence

from .selector import ToolSelector


def _default_messages_getter(request: Any) -> Any:
    """
    Best-effort extraction of messages from a LangChain model request.
    We keep this very defensive to survive LangChain internal changes.

    Tries in order:
      - request.state["messages"]
      - request.inputs["messages"]
      - request.messages
      - request.prompt / request.input (as fallback)
    """
    st = getattr(request, "state", None)
    if isinstance(st, dict) and "messages" in st:
        return st["messages"]

    inputs = getattr(request, "inputs", None)
    if isinstance(inputs, dict) and "messages" in inputs:
        return inputs["messages"]

    if hasattr(request, "messages"):
        return getattr(request, "messages")

    # fallback (may be a string prompt)
    if hasattr(request, "prompt"):
        return getattr(request, "prompt")
    if hasattr(request, "input"):
        return getattr(request, "input")

    raise ValueError("Could not extract messages from request. Provide messages_getter=...")


def _default_session_id_getter(request: Any) -> Optional[str]:
    """
    Best-effort extraction of a session id from request/config.
    Common places:
      - request.config["configurable"]["thread_id"] (LangGraph-style)
      - request.config["configurable"]["session_id"]
      - request.config["thread_id"] / ["session_id"]
      - request.state["session_id"]
    """
    cfg = getattr(request, "config", None)
    if isinstance(cfg, dict):
        conf = cfg.get("configurable")
        if isinstance(conf, dict):
            for key in ("thread_id", "session_id"):
                v = conf.get(key)
                if isinstance(v, str) and v:
                    return v
        for key in ("thread_id", "session_id"):
            v = cfg.get(key)
            if isinstance(v, str) and v:
                return v

    st = getattr(request, "state", None)
    if isinstance(st, dict):
        v = st.get("session_id")
        if isinstance(v, str) and v:
            return v

    return None


def make_toolscope_tool_selection_middleware(
    *,
    selector: ToolSelector,
    all_tools: Sequence[Any],
    k: int = 12,
    allow_tags: Optional[Sequence[str]] = None,
    deny_tags: Optional[Sequence[str]] = None,
    session_id_getter: Optional[Callable[[Any], Optional[str]]] = None,
    messages_getter: Optional[Callable[[Any], Any]] = None,
    # observability
    trace: bool = False,
    trace_sink: Optional[Any] = None,
) -> Any:
    """
    Create a LangChain Agent middleware that uses ToolScope to dynamically select
    a subset of tools per LLM call, without changing tool semantics.

    This integrates correctly with `langchain.agents.create_agent(...)` by
    overriding the tools list on each model call via request.override(tools=...).

    Notes:
      - Does NOT require langchain to be installed until this function is called.
      - If LangChain changes internal request shapes, pass messages_getter/session_id_getter.
    """
    # Import lazily to avoid forcing langchain dependency.
    try:
        from langchain.agents.middleware import wrap_model_call  # type: ignore
    except Exception as e:
        raise ImportError(
            "LangChain is required for ToolScope middleware. Install e.g.:\n"
            "  pip install langchain langgraph\n"
        ) from e

    mg = messages_getter or _default_messages_getter
    sg = session_id_getter or _default_session_id_getter

    @wrap_model_call
    def _mw(request: Any, handler: Callable[[Any], Any]) -> Any:
        messages = mg(request)
        session_id = sg(request)

        if trace and hasattr(selector, "select_with_trace"):
            filtered_tools, tr = selector.select_with_trace(
                messages,
                all_tools,
                k=k,
                allow_tags=allow_tags,
                deny_tags=deny_tags,
                session_id=session_id,
                turn_id=None,
                trace_sink=trace_sink,
            )
            # Attach trace to request if there's a place for it (best-effort).
            # We do NOT rely on this anywhere.
            try:
                meta = getattr(request, "metadata", None)
                if isinstance(meta, dict):
                    meta["toolscope_trace"] = tr
            except Exception:
                pass
        else:
            filtered_tools = selector.select(
                messages,
                all_tools,
                k=k,
                allow_tags=allow_tags,
                deny_tags=deny_tags,
                session_id=session_id,
            )

        # Override tools for this model call.
        if not hasattr(request, "override"):
            raise AttributeError(
                "LangChain middleware expected request.override(tools=...). "
                "LangChain request API may have changed."
            )
        new_request = request.override(tools=filtered_tools)
        return handler(new_request)

    return _mw
