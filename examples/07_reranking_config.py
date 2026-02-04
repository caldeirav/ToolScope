import toolscope

TOOLS = [
    {"type":"function","function":{"name":"jira_create_issue","description":"Create a Jira issue.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"jira_search_issues","description":"Search Jira issues by query.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"confluence_search","description":"Search Confluence pages.","parameters":{"type":"object","properties":{}}}},
]

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("jira")), float(sum(map(ord, t)) % 101)] for t in texts]


if __name__ == "__main__":
    # Reranking is opt-in.
    # This is the config shape; running it requires sentence-transformers cross-encoder
    # OR you inject a custom reranker (see tests for how).
    reranking = toolscope.RerankingConfig(
        provider="sentence-transformers",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size=10,
        allow_download=False,
    )

    idx = toolscope.index(TOOLS, embedder=TinyDummyEmbedder(), reranking=reranking)
    # If the model isn't present locally and allow_download=False, this will throw.
    try:
        out = idx.filter("create jira ticket", k=2)
        print([t["function"]["name"] for t in out])
    except Exception as e:
        print("Reranking model not available locally (expected if not installed/preloaded).")
        print("Error:", e)
