import toolscope


def test_allow_tags_filters_results(openai_tools, embedder):
    idx = toolscope.index(openai_tools, embedder=embedder, normalizer=toolscope.AutoToolNormalizer())

    tools = idx.filter("create a ticket", k=10, allow_tags=["jira"])
    names = [t["function"]["name"] for t in tools]
    assert "jira_create_issue" in names
    assert "confluence_search" not in names


def test_deny_tags_filters_results(openai_tools, embedder):
    idx = toolscope.index(openai_tools, embedder=embedder, normalizer=toolscope.AutoToolNormalizer())

    tools = idx.filter("delete production", k=10, deny_tags=["dangerous"])
    names = [t["function"]["name"] for t in tools]
    assert "dangerous_delete_prod" not in names


def test_allow_and_deny_together(openai_tools, embedder):
    idx = toolscope.index(openai_tools, embedder=embedder, normalizer=toolscope.AutoToolNormalizer())

    tools = idx.filter("ticket for prod", k=10, allow_tags=["tickets"], deny_tags=["dangerous"])
    names = [t["function"]["name"] for t in tools]
    assert "jira_create_issue" in names
    assert "dangerous_delete_prod" not in names
