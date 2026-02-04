from toolscope.adapters.langchain import LangChainToolNormalizer, ToolSelector


class DummyTool:
    def __init__(self, name: str, description: str, tags=None, args_schema=None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.args_schema = args_schema


class DummyEmbedder:
    def embed_texts(self, texts):
        # deterministic small vectors
        out = []
        for t in texts:
            s = t or ""
            out.append([float(len(s) % 97), float(s.count("jira")), float(sum(map(ord, s)) % 101)])
        return out


def test_langchain_normalizer_roundtrip():
    tools = [DummyTool("jira_create_issue", "Create a Jira issue", tags=["jira", "tickets"])]
    n = LangChainToolNormalizer()
    canon = n.normalize(tools)
    assert canon[0].name == "jira_create_issue"
    assert "jira" in canon[0].tags
    assert canon[0].fingerprint is not None
    back = n.denormalize(canon)
    assert back == tools


def test_selector_filters_tools_allow_deny():
    tools = [
        DummyTool("jira_create_issue", "Create a Jira issue", tags=["jira", "tickets"]),
        DummyTool("dangerous_delete_prod", "Delete production data", tags=["dangerous", "prod"]),
        DummyTool("confluence_search", "Search Confluence pages", tags=["docs", "search"]),
    ]

    sel = ToolSelector(embedder=DummyEmbedder())
    out = sel.select("create jira ticket", tools, k=3, allow_tags=["jira"], deny_tags=["dangerous"])
    names = [t.name for t in out]
    assert "jira_create_issue" in names
    assert "dangerous_delete_prod" not in names


def test_selector_with_trace():
    tools = [
        DummyTool("jira_create_issue", "Create a Jira issue", tags=["jira"]),
        DummyTool("confluence_search", "Search Confluence pages", tags=["docs"]),
    ]
    sel = ToolSelector(embedder=DummyEmbedder())

    out, trace = sel.select_with_trace(
        "create jira ticket",
        tools,
        k=1,
        allow_tags=["jira"],
        session_id="s1",
        turn_id="t1",
    )

    assert len(out) == 1
    assert trace is not None
    assert trace.returned_tools == 1
    assert "jira" in trace.allow_tags
    assert trace.ms_total >= 0.0
