import toolscope


def test_public_exports_exist():
    # entrypoints
    assert callable(toolscope.filter)
    assert callable(toolscope.index)

    # configs
    assert hasattr(toolscope, "EmbeddingConfig")
    assert hasattr(toolscope, "ToolTextConfig")
    assert hasattr(toolscope, "RerankingConfig")

    # optional observability API
    assert hasattr(toolscope, "filter_with_trace")
    assert hasattr(toolscope, "ToolScopeTrace")

    # normalizers
    assert hasattr(toolscope, "AutoToolNormalizer")
    assert hasattr(toolscope, "McpToolNormalizer")
    assert hasattr(toolscope, "OpenAIToolDictNormalizer")
