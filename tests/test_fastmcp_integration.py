import pytest

from toolscope.adapters.fastmcp import ToolScopeFastMCPClient, FastMCPWrapperConfig


pytestmark = pytest.mark.integration


class DummyEmbedder:
    def embed_texts(self, texts):
        out = []
        for t in texts:
            s = t or ""
            # deterministic but not semantic; good enough for plumbing/integration
            out.append([float(len(s) % 97), float(s.count("add") + s.count("multiply")), float(sum(map(ord, s)) % 101)])
        return out


@pytest.mark.asyncio
async def test_fastmcp_inmemory_client_roundtrip():
    fastmcp = pytest.importorskip("fastmcp")

    FastMCP = fastmcp.FastMCP
    Client = fastmcp.Client

    # ---- create in-memory server
    server = FastMCP("ToolScopeTestServer")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # ---- connect in-memory client
    async with Client(server) as upstream:
        wrapper = ToolScopeFastMCPClient(
            upstream,
            config=FastMCPWrapperConfig(
                embedder=DummyEmbedder(),
                auto_refresh=False,  # deterministic; no need to diff on each call
            ),
        )

        # list filtered tools
        tools = await wrapper.list_tools(
            [{"role": "user", "content": "please multiply numbers"}],
            k=1,
        )
        assert isinstance(tools, list)
        assert len(tools) == 1

        # tool objects from FastMCP commonly expose `.name`
        # (but to be robust, allow dict shape too)
        name = getattr(tools[0], "name", None) or tools[0].get("name")
        assert name in {"multiply", "add"}

        # call a tool through wrapper (passthrough)
        res = await wrapper.call_tool("add", {"a": 2, "b": 3})

        # FastMCP tool results generally expose `.data`
        # but may also expose `.content` depending on version
        data = getattr(res, "data", None)
        if data is None:
            # fallback: some versions return content list
            content = getattr(res, "content", None)
            if content and hasattr(content[0], "text"):
                data = int(content[0].text)
        assert data == 5
