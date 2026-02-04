import toolscope


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "very_long_tool",
            "description": "A" * 1000 + " This is the real meaning at the end.",
            "parameters": {"type": "object", "properties": {}},
        }
    }
]

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]


if __name__ == "__main__":
    cfg = toolscope.ToolTextConfig(
        fields=("name", "description"),
        truncate=256,
        preprocessors=(toolscope.collapse_whitespace(),),
    )

    idx = toolscope.index(TOOLS, embedder=TinyDummyEmbedder(), tool_text_config=cfg)
    tools = idx.filter("real meaning", k=1)
    print("Selected:", tools[0]["function"]["name"])
