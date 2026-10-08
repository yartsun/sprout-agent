"""Tiny MCP server used by the tests."""
import base64
import io

from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

server = FastMCP("fake")


@server.tool()
def shout(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


@server.tool()
def swatch() -> Image:
    """Return a small image."""
    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), (200, 50, 50)).save(buffer, format="PNG")
    return Image(data=buffer.getvalue(), format="png")


@server.tool()
def fail() -> str:
    """Always fails."""
    raise RuntimeError("broken on purpose " + base64.b64encode(b"x").decode())


if __name__ == "__main__":
    server.run()
