import toolscope

TOOLS = [
    {"type":"function","function":{"name":"jira_create_issue","description":"Create a Jira issue.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"confluence_search","description":"Search Confluence pages.","parameters":{"type":"object","properties":{}}}},
]

if __name__ == "__main__":
    # Example for a real production setup where you already have an embedding service.
    # Your service should accept POST {endpoint} with JSON:
    #   {"texts": [...], "model": "..."}   (model optional)
    # and respond with:
    #   {"vectors": [[...], [...]]}
    embedding = toolscope.EmbeddingConfig(
        provider="http",
        endpoint="http://localhost:8080/embed",
        model="my-embeddings-v1",
    )

    # This will call your HTTP service. Uncomment when you have it running.
    # idx = toolscope.index(TOOLS, embedding=embedding)
    # out = idx.filter("search docs", k=1)
    # print([t["function"]["name"] for t in out])

    print("This example is a wiring template. Start an embedding service at localhost:8080 to run it.")
