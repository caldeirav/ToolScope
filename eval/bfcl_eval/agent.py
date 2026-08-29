"""
LangGraph one-turn paper agent.

select_tools lives in the eval loop (needed for Recall/NDCG). This module
only runs ``llm_call``: bind already-selected LangChain tools and read
``AIMessage.tool_calls``. Tools are never executed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .model import ParsedToolCall, _extract_tool_call
from .tools import catalog_to_langchain, dedupe_lc_tools_by_safe_name, original_tool_name


@dataclass
class PredictResult:
    raw: str
    predicted: Optional[ParsedToolCall]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float = 0.0
    native_tools: bool = True
    error: Optional[str] = None


def _to_lc_messages(messages: List[Dict]) -> List[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out: List[Any] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _as_lc_tools(tools: Sequence[Any]) -> List[Any]:
    openai_dicts = []
    lc = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function":
            openai_dicts.append(t)
        else:
            lc.append(t)
    if openai_dicts:
        lc.extend(catalog_to_langchain(openai_dicts))
    return dedupe_lc_tools_by_safe_name(lc)


def _usage_tokens(ai: Any) -> tuple[Optional[int], Optional[int]]:
    meta = getattr(ai, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        inp = meta.get("input_tokens") or meta.get("prompt_tokens")
        out = meta.get("output_tokens") or meta.get("completion_tokens")
        return (
            int(inp) if inp is not None else None,
            int(out) if out is not None else None,
        )
    resp_meta = getattr(ai, "response_metadata", None) or {}
    if isinstance(resp_meta, dict):
        usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
        if isinstance(usage, dict):
            inp = usage.get("prompt_tokens") or usage.get("input_tokens")
            out = usage.get("completion_tokens") or usage.get("output_tokens")
            return (
                int(inp) if inp is not None else None,
                int(out) if out is not None else None,
            )
    return None, None


def _parse_ai_message(ai: Any, lc_tools: Sequence[Any]) -> tuple[str, Optional[ParsedToolCall]]:
    safe_to_orig = {getattr(t, "name", ""): original_tool_name(t) for t in lc_tools}
    tool_calls = getattr(ai, "tool_calls", None) or []
    if tool_calls:
        tc = tool_calls[0]
        if isinstance(tc, dict):
            name = tc.get("name") or ""
            args = tc.get("args") or tc.get("arguments") or {}
        else:
            name = getattr(tc, "name", "") or ""
            args = getattr(tc, "args", None) or {}
        if not isinstance(args, dict):
            args = {}
        orig = safe_to_orig.get(name, name)
        raw = json.dumps({"name": orig, "arguments": args}, ensure_ascii=False)
        return raw, ParsedToolCall(name=orig, arguments=args)

    content = getattr(ai, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    parsed = _extract_tool_call(content)
    if parsed is not None:
        parsed.name = safe_to_orig.get(parsed.name, parsed.name)
    return content, parsed


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (429, 500, 502, 503, 529):
        return True
    text = str(exc).lower()
    return any(s in text for s in ("429", "rate limit", "timeout", "temporar", "unavailable"))


def _invoke_with_retry(bound: Any, messages: List[Any], attempts: int = 5) -> Any:
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    @retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _call() -> Any:
        return bound.invoke(messages)

    return _call()


def _env_nonempty(name: str) -> str:
    return (os.environ.get(name) or "").strip().strip('"').strip("'")


def require_llm_credentials(entry: Dict[str, Any]) -> None:
    """Fail fast if the provider's API key is missing.

    OpenAI-compatible hosts use ``OPENAI_API_KEY`` (or ``api_key_env``).
    Google uses ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``.
    """
    provider = (entry.get("provider") or "openai").lower()
    if provider in ("google", "gemini"):
        if not (
            _env_nonempty("GOOGLE_API_KEY")
            or _env_nonempty("GEMINI_API_KEY")
            or str(entry.get("api_key") or "").strip().strip('"').strip("'")
        ):
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")
        return
    key = str(entry.get("api_key") or "").strip().strip('"').strip("'")
    env_name = str(entry.get("api_key_env") or "OPENAI_API_KEY")
    if not key:
        key = _env_nonempty(env_name) or _env_nonempty("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            f"API key missing for {entry.get('name')}. "
            f"Set {env_name} in the repo-root .env."
        )


def make_llm(entry: Dict[str, Any]) -> Any:
    """Build ChatOpenAI (OpenAI-compatible) or ChatGoogleGenerativeAI from a model entry."""
    require_llm_credentials(entry)
    provider = (entry.get("provider") or "openai").lower()
    model_name = entry["name"]
    max_tokens = int(entry.get("max_new_tokens") or 512)
    api_key = str(entry.get("api_key") or "").strip().strip('"').strip("'")

    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or _env_nonempty("GOOGLE_API_KEY") or _env_nonempty("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=0,
            max_output_tokens=max_tokens,
        )

    from langchain_openai import ChatOpenAI

    env_name = str(entry.get("api_key_env") or "OPENAI_API_KEY")
    api_key = api_key or _env_nonempty(env_name) or _env_nonempty("OPENAI_API_KEY")
    base_url = (
        str(entry.get("base_url") or "").strip()
        or _env_nonempty("OPENAI_BASE_URL")
        or None
    )
    if not api_key:
        raise RuntimeError(
            f"API key missing for {model_name}. Set {env_name} in the repo-root .env."
        )
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
    )


class LangGraphAgent:
    """
    One-turn LangGraph agent: START → llm_call → END.

    ``predict(messages, tools)`` binds ``tools`` (already selected) and returns
    the first tool call. Fail-closed if bind_tools is rejected.
    """

    def __init__(self, entry: Dict[str, Any]) -> None:
        self.entry = entry
        self.model_name = entry["name"]
        self._llm: Any = None
        self._graph: Any = None
        self._bind_ok = False

    def load(self) -> None:
        from langgraph.graph import END, START, StateGraph
        from typing import TypedDict

        self._llm = make_llm(self.entry)

        class AgentState(TypedDict):
            lc_messages: list
            selected_tools: list
            ai_message: Any

        llm = self._llm

        def llm_call(state: AgentState) -> dict:
            tools = state["selected_tools"]
            bound = llm.bind_tools(tools)
            ai = _invoke_with_retry(bound, state["lc_messages"])
            return {"ai_message": ai}

        builder = StateGraph(AgentState)
        builder.add_node("llm_call", llm_call)
        builder.add_edge(START, "llm_call")
        builder.add_edge("llm_call", END)
        self._graph = builder.compile()

        probe = catalog_to_langchain([{
            "type": "function",
            "function": {
                "name": "probe_tool",
                "description": "Connectivity probe. Do not call.",
                "parameters": {"type": "object", "properties": {}},
            },
        }])
        try:
            self._llm.bind_tools(probe)
            self._bind_ok = True
        except Exception as exc:
            raise RuntimeError(
                f"bind_tools rejected for {self.model_name}: {type(exc).__name__}: {exc}\n"
                "Paper protocol requires native tool calling; refusing prompt fallback."
            ) from exc

        print(
            f"  LangGraph agent ready: {self.model_name}  "
            f"(provider={self.entry.get('provider', 'openai')})"
        )

    def predict(self, messages: List[Dict], tools: Sequence[Any]) -> PredictResult:
        if self._graph is None:
            self.load()
        lc_tools = _as_lc_tools(tools)
        if not lc_tools:
            return PredictResult(raw="", predicted=None, native_tools=True)

        lc_messages = _to_lc_messages(messages)
        t0 = time.monotonic()
        try:
            state = self._graph.invoke({
                "lc_messages": lc_messages,
                "selected_tools": lc_tools,
                "ai_message": None,
            })
        except Exception:
            # Fail-closed for this condition only: no prompt fallback, no raise.
            return PredictResult(
                raw="",
                predicted=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                native_tools=True,
                error="api_fail",
            )
        latency_ms = (time.monotonic() - t0) * 1000
        ai = state.get("ai_message")
        raw, predicted = _parse_ai_message(ai, lc_tools)
        prompt_tok, completion_tok = _usage_tokens(ai)
        return PredictResult(
            raw=raw,
            predicted=predicted,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            latency_ms=latency_ms,
            native_tools=True,
        )

    def parse_tool_call(self, raw: str) -> Optional[ParsedToolCall]:
        return _extract_tool_call(raw)
