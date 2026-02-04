import toolscope


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "jira_create_issue",
            "description": "Create a Jira issue.",
            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
        "toolscope_tags": ["jira", "tickets"],
    },
    {
        "type": "function",
        "function": {
            "name": "confluence_search",
            "description": "Search Confluence pages.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
        "toolscope_tags": ["docs", "search"],
    },
]


# You must provide either `embedder=...` or `embedding=EmbeddingConfig(...)`.
# For a real setup, see examples/05_embedding_http_provider.py
class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("a")), float(sum(map(ord, t)) % 101)] for t in texts]


if __name__ == "__main__":
    filtered = toolscope.filter(
        "Create a ticket about build failures",
        TOOLS,
        k=1,
        embedder=TinyDummyEmbedder(),
    )
    print("Selected tools:", [t["function"]["name"] for t in filtered])
