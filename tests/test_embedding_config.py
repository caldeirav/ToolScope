import pytest
import toolscope


def test_make_index_rejects_embedder_and_embedding(openai_tools, embedder):
    with pytest.raises(ValueError):
        toolscope.index(
            openai_tools,
            embedder=embedder,
            embedding=toolscope.EmbeddingConfig(provider="http", endpoint="http://localhost:1/embed"),
        )


def test_filter_rejects_embedder_and_embedding(openai_tools, embedder):
    with pytest.raises(ValueError):
        toolscope.filter(
            "hello",
            openai_tools,
            embedder=embedder,
            embedding=toolscope.EmbeddingConfig(provider="http", endpoint="http://localhost:1/embed"),
        )
