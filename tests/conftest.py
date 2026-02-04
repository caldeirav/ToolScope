import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


class DummyEmbedder:
    """
    Deterministic embedder:
      - returns a 3D vector based on simple string features
      - normalized-ish but doesn't have to be perfect
    """
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for t in texts:
            s = t or ""
            # crude signals
            out.append([
                float(len(s) % 101),
                float(sum(ord(c) for c in s) % 251),
                float(s.count("a") + s.count("A")),
            ])
        return out


class DummyReranker:
    """
    Deterministic reranker:
      scores doc by overlap with query tokens.
    """
    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        q = set(query.lower().split())
        scores = []
        for d in docs:
            dd = set((d or "").lower().split())
            scores.append(float(len(q & dd)))
        return scores


@dataclass
class McpToolObj:
    name: str
    description: str
    inputSchema: Dict[str, Any]
    tags: List[str] | None = None


@pytest.fixture
def openai_tools():
    # OpenAI style tool dicts with ToolScope tags
    return [
        {
            "type": "function",
            "function": {
                "name": "jira_create_issue",
                "description": "Create a Jira issue in project ABC.",
                "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
            },
            "toolscope_tags": ["jira", "tickets"],
        },
        {
            "type": "function",
            "function": {
                "name": "confluence_search",
                "description": "Search Confluence pages by query.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
            "toolscope_tags": ["docs", "search"],
        },
        {
            "type": "function",
            "function": {
                "name": "dangerous_delete_prod",
                "description": "Delete production data. Extremely dangerous.",
                "parameters": {"type": "object", "properties": {"confirm": {"type": "boolean"}}},
            },
            "toolscope_tags": ["dangerous", "prod"],
        },
    ]


@pytest.fixture
def mcp_tools_dict():
    return [
        {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
            "toolscope_tags": ["comms"],
        },
        {
            "name": "calendar_create_event",
            "description": "Create a calendar event.",
            "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
            "annotations": {"tags": ["calendar"]},
        },
    ]


@pytest.fixture
def mcp_tools_obj():
    return [
        McpToolObj(
            name="github_create_issue",
            description="Create a GitHub issue in a repo.",
            inputSchema={"type": "object", "properties": {"repo": {"type": "string"}}},
            tags=["github", "tickets"],
        )
    ]


@pytest.fixture
def embedder():
    return DummyEmbedder()


@pytest.fixture
def reranker():
    return DummyReranker()
