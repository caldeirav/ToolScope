#!/usr/bin/env python3
"""
Minimal bind_tools smoke test against a running llama-server (/v1).

Exits 0 when the model returns a native tool call; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test tool calling via OpenAI-compatible /v1")
    parser.add_argument("--model", required=True, help="Model alias (llama-server -a)")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "local"),
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    try:
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("error: install eval/paper/requirements.txt (langchain-openai)", file=sys.stderr)
        return 2

    @tool
    def get_weather(city: str) -> str:
        """Return weather for a city."""
        return f"sunny in {city}"

    llm = ChatOpenAI(
        model=args.model,
        base_url=base_url,
        api_key=args.api_key or "local",
        temperature=0,
        max_tokens=128,
        timeout=float(os.environ.get("SMOKE_TIMEOUT_SECONDS", "120")),
    )

    bound = llm.bind_tools([get_weather])
    retries = int(os.environ.get("SMOKE_LOAD_RETRIES", "24"))
    delay = float(os.environ.get("SMOKE_LOAD_DELAY_SECONDS", "15"))
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ai = bound.invoke("What is the weather in Paris? Call the tool.")
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "503" in msg and "Loading model" in msg and attempt < retries:
                print(f"waiting for model load ({attempt}/{retries}) ...", file=sys.stderr)
                time.sleep(delay)
                continue
            raise
    else:
        raise last_err  # type: ignore[misc]
    tool_calls = getattr(ai, "tool_calls", None) or []
    if not tool_calls:
        content = getattr(ai, "content", "")
        print("FAIL: no tool_calls in response")
        print(f"content: {content!r}")
        return 1

    tc = tool_calls[0]
    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
    print(f"OK: tool_call name={name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
