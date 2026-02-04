import toolscope


TOOLS = [
    {
        "type": "function",
        "function": {"name": "jira_create_issue", "description": "Create a Jira issue.", "parameters": {"type":"object","properties":{}}},
        "toolscope_tags": ["jira", "tickets"],
    },
    {
        "type": "function",
        "function": {"name": "dangerous_delete_prod", "description": "Delete production data.", "parameters": {"type":"object","properties":{}}},
        "toolscope_tags": ["dangerous", "prod"],
    },
]

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("e")), float(sum(map(ord, t)) % 101)] for t in texts]


if __name__ == "__main__":
    idx = toolscope.index(TOOLS, embedder=TinyDummyEmbedder())

    safe = idx.filter("delete prod", k=5, deny_tags=["dangerous"])
    print("Safe tools:", [t["function"]["name"] for t in safe])

    only_jira = idx.filter("create ticket", k=5, allow_tags=["jira"])
    print("Jira-only:", [t["function"]["name"] for t in only_jira])
