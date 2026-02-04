import toolscope


TOOLS = [
    {"type":"function","function":{"name":"jira_create_issue","description":"Create a Jira issue.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["jira","tickets"]},
    {"type":"function","function":{"name":"jira_search_issues","description":"Search Jira issues.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["jira"]},
    {"type":"function","function":{"name":"confluence_search","description":"Search Confluence pages.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["docs","search"]},
    {"type":"function","function":{"name":"dangerous_delete_prod","description":"Delete production data.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["dangerous","prod"]},
]

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("jira") + t.count("confluence")), float(sum(map(ord, t)) % 101)] for t in texts]


if __name__ == "__main__":
    idx = toolscope.index(TOOLS, embedder=TinyDummyEmbedder())

    session_id = "agent-session-1"
    messages = []

    def step(user_msg: str):
        messages.append({"role": "user", "content": user_msg})
        tools, trace = idx.filter_with_trace(
            messages,
            k=3,
            session_id=session_id,
            allow_tags=["jira", "docs", "search", "tickets"],  # example allowlist
            deny_tags=["dangerous"],                           # hard safety denylist
        )
        names = [t["function"]["name"] for t in tools]
        print(f"\nUser: {user_msg}")
        print("Selected tools:", names)
        print(f"Trace: mode={trace.mode} ms_total={trace.ms_total:.2f} candidates={trace.retrieved_candidates}")

    step("We need to open a ticket about login failures")
    step("Add details: it happens after the recent deployment")
    step("Also search docs for the rollback procedure")
