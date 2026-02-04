import toolscope

TOOLS = [
    {"type":"function","function":{"name":"jira_create_issue","description":"Create a Jira issue.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["jira"]},
    {"type":"function","function":{"name":"jira_search_issues","description":"Search Jira issues.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["jira"]},
    {"type":"function","function":{"name":"confluence_search","description":"Search Confluence pages.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["docs"]},
]

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("jira")), float(sum(map(ord, t)) % 101)] for t in texts]


if __name__ == "__main__":
    session_cfg = toolscope.StickySessionConfig(
        enabled=True,
        similarity_threshold_reuse=0.95,
        similarity_threshold_refresh=0.80,
        sticky_keep=2,
        sticky_boost=0.03,
        ttl_seconds=3600,
    )

    idx = toolscope.index(TOOLS, embedder=TinyDummyEmbedder(), session_cfg=session_cfg)

    sid = "demo-session"

    out1 = idx.filter("create a jira ticket", k=2, session_id=sid)
    out2 = idx.filter("create a jira ticket for build failure", k=2, session_id=sid)  # similar => reuse/refresh

    print("Turn1:", [t["function"]["name"] for t in out1])
    print("Turn2:", [t["function"]["name"] for t in out2])
