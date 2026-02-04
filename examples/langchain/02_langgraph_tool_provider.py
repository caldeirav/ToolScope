from toolscope.adapters.langchain import ToolSelector, tool_provider


class DummyTool:
    def __init__(self, name: str, description: str, tags=None):
        self.name = name
        self.description = description
        self.tags = tags or []


class TinyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("jira")), float(sum(map(ord, t)) % 101)] for t in texts]


TOOLS = [
    DummyTool("jira_create_issue", "Create a Jira issue.", tags=["jira"]),
    DummyTool("jira_search_issues", "Search Jira issues.", tags=["jira"]),
    DummyTool("confluence_search", "Search Confluence docs.", tags=["docs"]),
]


if __name__ == "__main__":
    selector = ToolSelector(embedder=TinyEmbedder())

    provider = tool_provider(
        selector,
        TOOLS,
        k=2,
        allow_tags=["jira", "docs"],
        session_id_getter=lambda state: state.get("session_id"),
        messages_getter=lambda state: state["messages"],
    )

    # This mimics a LangGraph state object
    state = {
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Create a Jira ticket"}],
    }

    filtered_tools = provider(state)
    print("Tools for this turn:", [t.name for t in filtered_tools])
