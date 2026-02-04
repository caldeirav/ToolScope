from toolscope.adapters.langchain import ToolSelector


class DummyTool:
    def __init__(self, name: str, description: str, tags=None):
        self.name = name
        self.description = description
        self.tags = tags or []


class TinyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("jira")), float(sum(map(ord, t)) % 101)] for t in texts]


TOOLS = [
    DummyTool("jira_create_issue", "Create a Jira issue.", tags=["jira", "tickets"]),
    DummyTool("confluence_search", "Search Confluence pages.", tags=["docs", "search"]),
    DummyTool("dangerous_delete_prod", "Delete prod data.", tags=["dangerous"]),
]


if __name__ == "__main__":
    selector = ToolSelector(embedder=TinyEmbedder())

    messages = [{"role": "user", "content": "Open a ticket about failing builds"}]
    filtered = selector.select(
        messages,
        TOOLS,
        k=2,
        allow_tags=["jira", "tickets"],
        session_id="demo-session",
    )

    print("Selected tools:", [t.name for t in filtered])
