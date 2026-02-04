import asyncio
from toolscope.adapters.fastmcp import ToolScopeFastMCPClient, FastMCPWrapperConfig


class DummyEmbedder:
    def embed_texts(self, texts):
        out = []
        for t in texts:
            s = t or ""
            out.append([float(len(s) % 97), float(s.count("email")), float(sum(map(ord, s)) % 101)])
        return out


class FakeFastMCPClient:
    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name=None, arguments=None, **kwargs):
        self.calls.append((name, arguments))
        return {"ok": True, "name": name, "arguments": arguments}

    def set_tools(self, tools):
        self._tools = tools


def _tool(name, desc, tags=None):
    # A dict “MCP-like” tool; AutoToolNormalizer should handle once MCP normalizer exists.
    t = {"name": name, "description": desc, "inputSchema": {"type": "object", "properties": {}}}
    if tags:
        t["toolscope_tags"] = list(tags)
    return t


def test_fastmcp_wrapper_filters_tools():
    tools = [
        _tool("send_email", "Send an email", tags=["comms"]),
        _tool("create_event", "Create a calendar event", tags=["calendar"]),
        _tool("dangerous_delete_prod", "Delete prod", tags=["dangerous"]),
    ]
    upstream = FakeFastMCPClient(tools)

    cfg = FastMCPWrapperConfig(embedder=DummyEmbedder(), auto_refresh=True)
    wrapper = ToolScopeFastMCPClient(upstream, config=cfg)

    async def run():
        filtered = await wrapper.list_tools(
            [{"role": "user", "content": "please email the customer"}],
            k=2,
            deny_tags=["dangerous"],
        )
        names = [t.get("name") for t in filtered]
        assert "dangerous_delete_prod" not in names

    asyncio.run(run())


def test_fastmcp_wrapper_call_tool_passthrough():
    tools = [_tool("send_email", "Send an email", tags=["comms"])]
    upstream = FakeFastMCPClient(tools)
    cfg = FastMCPWrapperConfig(embedder=DummyEmbedder())
    wrapper = ToolScopeFastMCPClient(upstream, config=cfg)

    async def run():
        _ = await wrapper.list_tools("email", k=1)  # ensure index
        res = await wrapper.call_tool("send_email", {"to": "a@b.com"})
        assert res["ok"] is True
        assert upstream.calls[-1][0] == "send_email"

    asyncio.run(run())


def test_fastmcp_mark_dirty_triggers_refresh():
    tools1 = [_tool("send_email", "Send an email", tags=["comms"])]
    tools2 = tools1 + [_tool("create_event", "Create event", tags=["calendar"])]

    upstream = FakeFastMCPClient(tools1)
    cfg = FastMCPWrapperConfig(embedder=DummyEmbedder(), auto_refresh=False)
    wrapper = ToolScopeFastMCPClient(upstream, config=cfg)

    async def run():
        out1 = await wrapper.list_tools("email", k=5)
        assert len(out1) == 1

        # change upstream tools, but wrapper won't refresh unless marked dirty (auto_refresh=False)
        upstream.set_tools(tools2)
        out2 = await wrapper.list_tools("calendar", k=5)
        assert len(out2) == 1  # still old index

        wrapper.mark_dirty()
        out3 = await wrapper.list_tools("calendar", k=5)
        assert len(out3) == 2

    asyncio.run(run())
