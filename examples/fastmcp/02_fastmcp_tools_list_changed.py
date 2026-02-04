import asyncio
from toolscope.adapters.fastmcp import ToolScopeFastMCPClient, FastMCPWrapperConfig, make_toolscope_message_handler

class DummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("calendar")), float(sum(map(ord, t)) % 101)] for t in texts]

# Example only: you would use a real fastmcp.Client with message_handler=...
class FakeFastMCPClient:
    def __init__(self):
        self._tools = [{"name":"send_email","description":"Send email","inputSchema":{"type":"object","properties":{}}}]

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name=None, arguments=None, **kwargs):
        return {"ok": True}

    # simulate tools changing
    def add_tool(self):
        self._tools.append({"name":"create_event","description":"Create event","inputSchema":{"type":"object","properties":{}}})

async def main():
    upstream = FakeFastMCPClient()

    wrapper = ToolScopeFastMCPClient(
        upstream,
        config=FastMCPWrapperConfig(embedder=DummyEmbedder(), auto_refresh=False),
    )

    # In real FastMCP:
    # handler = make_toolscope_message_handler(wrapper.mark_dirty)
    # upstream = fastmcp.Client(..., message_handler=handler)
    # wrapper = ToolScopeFastMCPClient(upstream, config=...)

    tools1 = await wrapper.list_tools("email", k=5)
    print("Turn1 tools:", [t["name"] for t in tools1])

    upstream.add_tool()
    wrapper.mark_dirty()  # this would come from tools/list_changed notification
    tools2 = await wrapper.list_tools("calendar", k=5)
    print("Turn2 tools:", [t["name"] for t in tools2])

if __name__ == "__main__":
    asyncio.run(main())
