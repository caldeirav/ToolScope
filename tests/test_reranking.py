import toolscope


def test_reranking_changes_top_result(openai_tools, embedder, reranker):
    idx = toolscope.index(openai_tools, embedder=embedder, normalizer=toolscope.AutoToolNormalizer())

    # Inject reranker directly (index supports it if you added reranker field; if not,
    # pass reranking config and monkeypatch resolver; see note below).
    idx.reranker = reranker
    idx.reranking_config = toolscope.RerankingConfig(provider="st", model="dummy", pool_size=3)

    # Query includes "confluence" so reranker should push confluence_search up
    tools = idx.filter("please confluence search pages", k=2)
    names = [t["function"]["name"] for t in tools]
    assert names[0] == "confluence_search"
