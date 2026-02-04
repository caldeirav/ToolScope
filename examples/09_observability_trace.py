import json
import toolscope


TOOLS = [
    {"type":"function","function":{"name":"jira_create_issue","description":"Create a Jira issue.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["jira"]},
    {"type":"function","function":{"name":"confluence_search","description":"Search Confluence pages.","parameters":{"type":"object","properties":{}}}, "toolscope_tags":["docs"]},
]

class TinyDummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("jira") + t.count("confluence")), float(sum(map(ord, t)) % 101)] for t in texts]


class PrintJsonSink:
    def emit(self, trace):
        # dataclasses -> dict-ish
        print("TRACE:", json.dumps(trace.__dict__, default=lambda o: o.__dict__, indent=2))


if __name__ == "__main__":
    idx = toolscope.index(TOOLS, embedder=TinyDummyEmbedder())

    tools, trace = idx.filter_with_trace(
        "search confluence",
        k=2,
        session_id="s1",
        turn_id="t1",
        trace_sink=PrintJsonSink(),
    )

    print("Tools:", [t["function"]["name"] for t in tools])
    print("Mode:", trace.mode, "total_ms:", trace.ms_total)
