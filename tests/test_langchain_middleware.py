import sys
import types

from toolscope.adapters.langchain import ToolSelector
from toolscope.adapters.langchain.middleware import make_toolscope_tool_selection_middleware


class DummyTool:
    def __init__(self, name: str, description: str, tags=None):
        self.name = name
        self.description = description
        self.tags = tags or []


class DummyEmbedder:
    def embed_texts(self, texts):
        out = []
        for t in texts:
            s = t or ""
            out.append([float(len(s) % 97), float(s.count("jira")), float(sum(map(ord, s)) % 101)])
        return out


def _install_fake_langchain_middleware():
    """
    Provide a minimal stand-in for:
      from langchain.agents.middleware import wrap_model_call
    """
    lc = types.ModuleType("langchain")
    agents = types.ModuleType("langchain.agents")
    mw = types.ModuleType("langchain.agents.middleware")

    def wrap_model_call(fn):
        # decorator that returns fn unchanged
        return fn

    mw.wrap_model_call = wrap_model_call

    sys.modules["langchain"] = lc
    sys.modules["langchain.agents"] = agents
    sys.modules["langchain.agents.middleware"] = mw


class FakeRequest:
    def __init__(self, *, state=None, tools=None, config=None):
        self.state = state or {}
        self.tools = tools
        self.config = config or {"configurable": {"session_id": "s1"}}
        self.metadata = {}

    def override(self, **kwargs):
        # return a shallow clone with overrides applied
        nr = FakeRequest(state=self.state, tools=self.tools, config=self.config)
        nr.metadata = dict(self.metadata)
        for k, v in kwargs.items():
            setattr(nr, k, v)
        return nr


def test_toolscope_middleware_overrides_tools(monkeypatch):
    _install_fake_langchain_middleware()

    tools = [
        DummyTool("jira_create_issue", "Create a Jira issue", tags=["jira"]),
        DummyTool("confluence_search", "Search docs", tags=["docs"]),
        DummyTool("dangerous_delete_prod", "Delete prod", tags=["dangerous"]),
    ]

    selector = ToolSelector(embedder=DummyEmbedder())

    mw = make_toolscope_tool_selection_middleware(
        selector=selector,
        all_tools=tools,
        k=1,
        deny_tags=["dangerous"],
    )

    req = FakeRequest(state={"messages": [{"role": "user", "content": "create jira ticket"}]}, tools=tools)

    # Handler returns the tools it received, so we can assert override happened.
    def handler(r):
        return r.tools

    filtered = mw(req, handler)
    assert isinstance(filtered, list)
    assert len(filtered) == 1
    assert filtered[0].name != "dangerous_delete_prod"


def test_toolscope_middleware_trace_attaches(monkeypatch):
    _install_fake_langchain_middleware()

    tools = [
        DummyTool("jira_create_issue", "Create a Jira issue", tags=["jira"]),
        DummyTool("confluence_search", "Search docs", tags=["docs"]),
    ]

    selector = ToolSelector(embedder=DummyEmbedder())

    mw = make_toolscope_tool_selection_middleware(
        selector=selector,
        all_tools=tools,
        k=1,
        allow_tags=["jira"],
        trace=True,
    )

    req = FakeRequest(state={"messages": [{"role": "user", "content": "create jira ticket"}]}, tools=tools)

    def handler(r):
        # Ensure toolscope trace was attached best-effort
        assert "toolscope_trace" in r.metadata
        return r.tools

    out = mw(req, handler)
    assert len(out) == 1
    assert out[0].name == "jira_create_issue"
