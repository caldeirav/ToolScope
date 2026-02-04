import asyncio
from toolscope.adapters.fastmcp import ToolScopeFastMCPClient, FastMCPWrapperConfig

class DummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("github") + t.count("jira")), float(sum(map(ord, t)) % 101)] for t in texts]

class FakeFastMCPClient:
    async def list_tools(self):
        # Multi-server clients often prefix tool names: "server.tool"
        return [
            {"name":"github.create_issue","description":"Create GitHub issue","inputSchema":{"type":"object","properties":{}}},
            {"name":"jira.create_issue","description":"Create Jira issue","inputSchema":{"type":"object","properties":{}}},
            {"name":"slack.send_message","description":"Send Slack message","inputSchema":{"type":"object","properties":{}}},
        ]

    async def call_tool(self, name=None, arguments=None, **kwargs):
        return {"ok": True, "name": name, "arguments": arguments}

async def main():
    wrapper = ToolScopeFastMCPClient(
        FakeFastMCPClient(),
        config=FastMCPWrapperConfig(embedder=DummyEmbedder()),
    )

    tools = await wrapper.list_tools(
        [{"role": "user", "content": "open a jira ticket"}],
        k=2,
    )
    print("Filtered:", [t["name"] for t in tools])

if __name__ == "__main__":
    asyncio.run(main())
