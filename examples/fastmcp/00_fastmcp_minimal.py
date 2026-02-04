import asyncio
from toolscope.adapters.fastmcp import ToolScopeFastMCPClient, FastMCPWrapperConfig

# Replace with a real FastMCP client:
# from fastmcp import Client
# upstream = Client(...)

class DummyEmbedder:
    def embed_texts(self, texts):
        return [[float(len(t) % 97), float(t.count("email")), float(sum(map(ord, t)) % 101)] for t in texts]

class FakeFastMCPClient:
    def __init__(self):
        self._tools = [
            {"name":"send_email","description":"Send an email","inputSchema":{"type":"object","properties":{}}},
            {"name":"create_event","description":"Create a calendar event","inputSchema":{"type":"object","properties":{}}},
            {"name":"dangerous_delete_prod","description":"Delete prod","inputSchema":{"type":"object","properties":{}}},
        ]

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name=None, arguments=None, **kwargs):
        return {"ok": True, "name": name, "arguments": arguments}

async def main():
    upstream = FakeFastMCPClient()

    wrapper = ToolScopeFastMCPClient(
        upstream,
        config=FastMCPWrapperConfig(embedder=DummyEmbedder()),
    )

    messages = [{"role": "user", "content": "please email the customer"}]
    tools = await wrapper.list_tools(messages, k=2, deny_tags=["dangerous"])
    print("Filtered tools:", [t["name"] for t in tools])

    res = await wrapper.call_tool("send_email", {"to": "a@b.com"})
    print("Tool call:", res)

if __name__ == "__main__":
    asyncio.run(main())
