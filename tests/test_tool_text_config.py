import toolscope


def test_tool_text_defaults_truncate_and_fields(openai_tools):
    n = toolscope.AutoToolNormalizer()
    canon = n.normalize(openai_tools[:1])[0]

    cfg = toolscope.ToolTextConfig()  # defaults: name+description, truncate=256, no preprocessors
    text = cfg.render(canon)
    assert "jira_create_issue" in text
    assert "Create a Jira issue" in text
    assert len(text) <= 256


def test_tool_text_includes_tags_when_requested(openai_tools):
    n = toolscope.AutoToolNormalizer()
    canon = n.normalize(openai_tools[:1])[0]

    cfg = toolscope.ToolTextConfig(fields=("name", "description", "tags"), truncate=500)
    text = cfg.render(canon)
    assert "tickets" in text


def test_tool_text_preprocessors(openai_tools):
    n = toolscope.AutoToolNormalizer()
    canon = n.normalize(openai_tools[:1])[0]

    cfg = toolscope.ToolTextConfig(
        fields=("name", "description"),
        truncate=500,
        preprocessors=(toolscope.lowercase(), toolscope.collapse_whitespace()),
    )
    text = cfg.render(canon)
    assert text == text.lower()
    assert "  " not in text
