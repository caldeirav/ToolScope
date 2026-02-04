from typing import Annotated, TypedDict, Literal
import operator

from toolscope.adapters.langchain import ToolSelector

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage

from langgraph.graph import StateGraph, START, END


# --------------------------
# 1) Define tools (LangChain tools)
# --------------------------

@tool
def jira_create_issue(title: str) -> str:
    """Create a Jira issue with the given title."""
    return f"(fake) created Jira issue: {title}"

@tool
def confluence_search(query: str) -> str:
    """Search Confluence pages by query."""
    return f"(fake) search results for: {query}"

@tool
def dangerous_delete_prod(confirm: bool) -> str:
    """Delete production data (dangerous)."""
    return "(fake) deleted prod data" if confirm else "refused"

ALL_TOOLS = [jira_create_issue, confluence_search, dangerous_delete_prod]


# Optional: attach tags so allow/deny works nicely
# LangChain @tool returns objects with `.tags` in some setups; to be safe we set them.
jira_create_issue.tags = ["jira", "tickets"]
confluence_search.tags = ["docs", "search"]
dangerous_delete_prod.tags = ["dangerous", "prod"]


# --------------------------
# 2) ToolScope selector
# --------------------------

class TinyDemoEmbedder:
    """
    Replace with toolscope.EmbeddingConfig(provider="http", ...) in production.
    This is just to keep the example runnable without extra services.
    """
    def embed_texts(self, texts):
        out = []
        for t in texts:
            s = t or ""
            out.append([float(len(s) % 97), float(s.count("jira") + s.count("confluence")), float(sum(map(ord, s)) % 101)])
        return out

selector = ToolSelector(
    embedder=TinyDemoEmbedder(),
    # Optional: sticky multi-turn behavior
    # session_cfg=toolscope.StickySessionConfig(enabled=True, similarity_threshold_reuse=0.95, similarity_threshold_refresh=0.80, sticky_keep=2),
)


# --------------------------
# 3) Define graph state
# --------------------------

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    selected_tools: list  # List[BaseTool]
    session_id: str


# --------------------------
# 4) Node: select tools for this turn (ToolScope)
# --------------------------

def select_tools_node(state: MessagesState) -> dict:
    messages = state["messages"]
    sid = state.get("session_id", "demo-session")

    # Filter tools based on the conversation so far
    filtered = selector.select(
        messages,
        ALL_TOOLS,
        k=2,
        session_id=sid,
    )

    return {"selected_tools": filtered}


# --------------------------
# 5) Node: call the model with selected tools bound
# --------------------------

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_call(state: MessagesState) -> dict:
    tools = state["selected_tools"]
    model_with_tools = model.bind_tools(tools)

    sys = SystemMessage(content="You are a helpful assistant. Use tools when needed.")
    ai_msg = model_with_tools.invoke([sys] + state["messages"])
    return {"messages": [ai_msg]}


# --------------------------
# 6) Node: execute tools requested by the model (only among selected tools)
# --------------------------

def tool_node(state: MessagesState) -> dict:
    tools = state["selected_tools"]
    tools_by_name = {t.name: t for t in tools}

    last = state["messages"][-1]
    results = []

    for tool_call in getattr(last, "tool_calls", []):
        name = tool_call["name"]
        args = tool_call.get("args", {})

        tool_obj = tools_by_name.get(name)
        if tool_obj is None:
            # Model tried to call a tool it doesn't have; treat as an observation.
            results.append(ToolMessage(content=f"Tool '{name}' is not available.", tool_call_id=tool_call["id"]))
            continue

        observation = tool_obj.invoke(args)
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": results}


# --------------------------
# 7) Routing logic: tool call or finish
# --------------------------

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return END


# --------------------------
# 8) Build graph: select_tools -> llm_call -> (tool_node loop) -> llm_call -> ...
# --------------------------

builder = StateGraph(MessagesState)
builder.add_node("select_tools", select_tools_node)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "llm_call")
builder.add_conditional_edges("llm_call", should_continue, {"tool_node": "tool_node", END: END})
builder.add_edge("tool_node", "select_tools")  # re-select tools after tool results (per-turn)

agent = builder.compile()


if __name__ == "__main__":
    prompt = "Create a Jira ticket titled 'Build failing on main' and then search Confluence for rollback steps."
    state = {
        "messages": [HumanMessage(content=prompt)],
        "selected_tools": [],
        "session_id": "demo-session-1",
    }

    final = agent.invoke(state)

    print("\n--- Final transcript ---")
    for m in final["messages"]:
        try:
            m.pretty_print()
        except Exception:
            print(m)
