import toolscope


OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "jira_create_issue",
        "description": "Create a Jira issue.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
    },
}

MCP_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient.",
    "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
}

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("a")), float(sum(map(ord, t)) % 101)] for t in texts]


if __name__ == "__main__":
    tools = [OPENAI_TOOL, MCP_TOOL]

    idx = toolscope.index(
        tools,
        embedder=TinyDummyEmbedder(),
        normalizer=toolscope.AutoToolNormalizer(),
    )

    out = idx.filter("email the customer", k=2)
    print("Returned tool shapes:", [type(t) for t in out])
    print("Names:", [(t.get("function", {}).get("name") if "function" in t else t.get("name")) for t in out])
