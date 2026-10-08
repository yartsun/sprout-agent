import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from sprout.config import Settings

PUBLIC_IP = "93.184.215.14"


def public_resolver(host, port, *args, **kwargs):
    """Every host resolves to a public address unless it is obviously local."""
    address = {"localhost": "127.0.0.1", "metadata.internal": "169.254.169.254"}.get(host, PUBLIC_IP)
    return [(2, 1, 6, "", (address, 0))]


def png_bytes(size=(2000, 1000), color=(30, 120, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(workspace=tmp_path / "workspace", mcp_config=tmp_path / "missing.json", max_page_chars=200)


def web_transport(pages: dict[str, httpx.Response]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        response = pages.get(str(request.url))
        if response is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(response.status_code, headers=response.headers, content=response.content)

    return httpx.MockTransport(handler)


class FakeLLM:
    """OpenAI-compatible endpoint that replays scripted assistant messages and records requests."""

    def __init__(self, replies: list[dict], status: int = 200):
        self.replies = list(replies)
        self.status = status
        self.requests: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "fake"}]})
            body = json.loads(request.content)
            self.requests.append(body)
            if self.status != 200:
                return httpx.Response(self.status, text="model exploded")
            return httpx.Response(200, json={"choices": [{"message": self.replies.pop(0)}]})

        return httpx.MockTransport(handler)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def answer(text: str) -> dict:
    return {"role": "assistant", "content": text}
