import pytest
import toolscope


def test_openai_normalizer_roundtrip(openai_tools):
    n = toolscope.OpenAIToolDictNormalizer()
    canon = n.normalize(openai_tools)
    assert len(canon) == len(openai_tools)
    assert all(c.fingerprint is not None for c in canon)
    back = n.denormalize(canon)
    assert back == openai_tools


def test_mcp_normalizer_dict_roundtrip(mcp_tools_dict):
    n = toolscope.McpToolNormalizer()
    canon = n.normalize(mcp_tools_dict)
    assert len(canon) == len(mcp_tools_dict)
    assert all(c.fingerprint is not None for c in canon)
    back = n.denormalize(canon)
    assert back == mcp_tools_dict


def test_mcp_normalizer_object_roundtrip(mcp_tools_obj):
    n = toolscope.McpToolNormalizer()
    canon = n.normalize(mcp_tools_obj)
    assert len(canon) == len(mcp_tools_obj)
    assert canon[0].name == "github_create_issue"
    assert "tickets" in canon[0].tags
    back = n.denormalize(canon)
    assert back == mcp_tools_obj


def test_auto_normalizer_mixed(openai_tools, mcp_tools_dict, mcp_tools_obj):
    n = toolscope.AutoToolNormalizer()
    mixed = [openai_tools[0], mcp_tools_dict[0], mcp_tools_obj[0]]
    canon = n.normalize(mixed)
    assert [c.name for c in canon] == ["jira_create_issue", "send_email", "github_create_issue"]
    back = n.denormalize(canon)
    assert back == mixed


def test_auto_normalizer_unknown_shape_raises():
    n = toolscope.AutoToolNormalizer()
    with pytest.raises(TypeError):
        n.normalize([{"not_a_tool": True}])
