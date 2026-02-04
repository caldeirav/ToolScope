import toolscope

TOOLS = [
    {"type":"function","function":{"name":"jira_create_issue","description":"Create a Jira issue.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"confluence_search","description":"Search Confluence pages.","parameters":{"type":"object","properties":{}}}},
]

if __name__ == "__main__":
    # Requires: pip install -e ".[st]"
    # By default allow_download=False, so you should pre-download the model or set allow_download=True.
    embedding = toolscope.EmbeddingConfig(
        provider="sentence-transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
        allow_download=False,
    )

    try:
        idx = toolscope.index(TOOLS, embedding=embedding)
        out = idx.filter("search confluence", k=1)
        print([t["function"]["name"] for t in out])
    except Exception as e:
        print("Failed (likely model not present locally).")
        print("Either pre-download the model or set allow_download=True.")
        print("Error:", e)
