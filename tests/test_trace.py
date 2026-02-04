import toolscope


def test_filter_with_trace(openai_tools, embedder):
    idx = toolscope.index(openai_tools, embedder=embedder, normalizer=toolscope.AutoToolNormalizer())

    tools, trace = idx.filter_with_trace(
        "create a jira ticket",
        k=2,
        allow_tags=["jira"],
        session_id="sess-123",
        turn_id="t1",
    )

    assert isinstance(tools, list)
    assert trace.returned_tools == len(tools)
    assert trace.ms_total >= 0.0
    assert trace.ms_embed_query >= 0.0
    assert trace.retrieved_candidates >= 2
    assert "jira" in trace.allow_tags

    # candidates include vector_score at least for some
    if trace.candidates:
        assert trace.candidates[0].vector_score is not None


def test_top_level_filter_with_trace(openai_tools, embedder):
    # Stateless end-to-end
    tools, trace = toolscope.filter_with_trace(
        "search confluence",
        openai_tools,
        k=2,
        embedder=embedder,
        normalizer=toolscope.AutoToolNormalizer(),
    )
    assert tools
    assert trace.ms_total >= 0.0
