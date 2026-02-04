from toolscope.adapters.langchain import ToolSelector, make_toolscope_tool_selection_middleware

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# --------------------------
# Tools
# --------------------------

@tool
def jira_create_issue(title: str) -> str:
    """Create a Jira issue with the given title."""
    return f"(fake) created Jira issue: {title}"

@tool
def confluence_search(query: str) -> str:
    """Search Confluence pages by query."""
    return f"(fake) confluence results for: {query}"

@tool
def dangerous_delete_prod(confirm: bool) -> str:
    """Delete production data (dangerous)."""
    return "(fake) deleted prod data" if confirm else "refused"

ALL_TOOLS = [jira_create_issue, confluence_search, dangerous_delete_prod]

# Add tags (ToolScope allow/deny uses these)
jira_create_issue.tags = ["jira", "tickets"]
confluence_search.tags = ["docs", "search"]
dangerous_delete_prod.tags = ["dangerous", "prod"]


# --------------------------
# ToolScope selector
# --------------------------

class TinyDemoEmbedder:
    """
    Replace with toolscope.EmbeddingConfig(provider="http", ...) in production.
    """
    def embed_texts(self, texts):
        out = []
        for t in texts:
            s = t or ""
            out.append([float(len(s) % 97), float(s.count("jira") + s.count("confluence")), float(sum(map(ord, s)) % 101)])
        return out

selector = ToolSelector(embedder=TinyDemoEmbedder())


# --------------------------
# Middleware: ToolScope selects tools per turn
# --------------------------

toolscope_mw = make_toolscope_tool_selection_middleware(
    selector=selector,
    all_tools=ALL_TOOLS,
    k=2,
    trace=True,               # optional observability
)


# --------------------------
# Create agent (full loop)
# --------------------------

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=model,
    tools=ALL_TOOLS,                 # register superset
    middleware=[toolscope_mw],        # dynamically override tools per model call
)

if __name__ == "__main__":
    # create_agent uses graph-style invocation: {"messages":[...]}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Create a Jira ticket titled 'Build failing on main' and search docs for rollback steps."}]},
        config={"configurable": {"session_id": "demo-session-1"}},
    )
    print(result)
