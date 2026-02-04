import toolscope


def test_toolscope_filter_end_to_end_openai(openai_tools, embedder):
    tools = toolscope.filter(
        "create a ticket",
        openai_tools,
        k=2,
        embedder=embedder,
        normalizer=toolscope.AutoToolNormalizer(),
        allow_tags=["tickets"],
        deny_tags=["dangerous"],
    )
    assert tools
    assert all(t["type"] == "function" for t in tools)


def test_toolscope_index_end_to_end_mixed(mcp_tools_dict, openai_tools, embedder):
    mixed = [mcp_tools_dict[0], openai_tools[0]]
    idx = toolscope.index(mixed, embedder=embedder, normalizer=toolscope.AutoToolNormalizer())

    tools = idx.filter("send email", k=2)
    # round-trip types: could return dict MCP tool and dict OpenAI tool
    assert len(tools) >= 1
