from typing import Any, Callable


def _import_fastmcp_message_handler_base():
    # Lazy import so toolscope does not depend on fastmcp at import time.
    try:
        from fastmcp.client.messages import MessageHandler  # type: ignore
        return MessageHandler
    except Exception as e:
        raise ImportError(
            "FastMCP is required for ToolScope FastMCP message handler.\n"
            "Install with: pip install fastmcp"
        ) from e


class ToolScopeMessageHandler(_import_fastmcp_message_handler_base()):
    """
    FastMCP message handler that listens for notifications/tools/list_changed
    and triggers a callback.

    Use it by passing to fastmcp.Client(..., message_handler=ToolScopeMessageHandler(cb)).
    """
    def __init__(self, on_tools_changed: Callable[[], None]) -> None:
        super().__init__()
        self._cb = on_tools_changed

    async def on_tool_list_changed(self, notification: Any) -> None:
        # Notification type may differ across versions; we only care about the event.
        self._cb()


def make_toolscope_message_handler(on_tools_changed: Callable[[], None]) -> ToolScopeMessageHandler:
    return ToolScopeMessageHandler(on_tools_changed)
