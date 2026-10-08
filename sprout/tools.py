"""Built-in tools: web search, page reading, image viewing and files in the workspace folder."""
from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import re
import socket
from collections.abc import Callable
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

from .config import Settings

Resolver = Callable[..., list]
ToolResult = tuple[str, "str | None"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept-Language": "en;q=0.9,*;q=0.5",
}


class BlockedURL(ValueError):
    """The URL points at this machine or a private network."""


def check_public_url(url: str, allow_private: bool = False, resolve: Resolver = socket.getaddrinfo) -> None:
    """Only http(s) to public addresses, so a page cannot steer the agent to localhost, the LAN or cloud metadata."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL(f"Only http(s) URLs are allowed: {url}")
    if allow_private:
        return
    try:
        addresses = {info[4][0] for info in resolve(parts.hostname, None)}
    except socket.gaierror as error:
        raise BlockedURL(f"Cannot resolve {parts.hostname}") from error
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise BlockedURL(f"{parts.hostname} resolves to a non-public address ({ip}); set ALLOW_PRIVATE_URLS=1 to allow")


def workspace_path(workspace: Path, relative: str) -> Path:
    """Resolve a path inside the workspace; symlinks and `..` cannot escape it."""
    workspace = workspace.resolve()
    path = (workspace / (relative or ".").lstrip("/")).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError("Path must stay inside the workspace folder")
    return path


def image_to_data_url(raw: bytes, max_side: int = 1024) -> str:
    image = Image.open(io.BytesIO(raw))
    if getattr(image, "is_animated", False):
        image.seek(0)
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


IMAGE_PATTERNS = [
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image',
    r'<img[^>]+(?:data-src|src)=["\']([^"\']+)',
]


def extract_images(html: str, base_url: str, limit: int = 12) -> list[str]:
    found: list[str] = []
    for pattern in IMAGE_PATTERNS:
        for match in re.findall(pattern, html, flags=re.I):
            url = urljoin(base_url, unescape(match.strip()))
            if url.startswith("http") and not url.lower().split("?")[0].endswith(".svg") and url not in found:
                found.append(url)
            if len(found) >= limit:
                return found
    return found


def html_to_text(html: str) -> str:
    import trafilatura

    text = trafilatura.extract(html, include_links=False, include_tables=True) or ""
    if not text:
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", unescape(text)).strip()
    return text


def duckduckgo(query: str, limit: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=limit))


def flag(value: Any) -> bool:
    """Tool arguments from small models often arrive as strings: "False" must not mean True."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _function(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


class Toolbox:
    """Built-in tools plus whatever the MCP hub exposes, behind one `call(name, args)`."""

    def __init__(
        self,
        settings: Settings,
        mcp: Any = None,
        transport: httpx.AsyncBaseTransport | None = None,
        search: Callable[[str, int], list[dict]] = duckduckgo,
        resolve: Resolver = socket.getaddrinfo,
    ):
        self.settings = settings
        self.mcp = mcp
        self.search = search
        self.resolve = resolve
        self.http = httpx.AsyncClient(transport=transport, headers=HEADERS, event_hooks={"request": [self._guard]})

    async def _guard(self, request: httpx.Request) -> None:
        # Runs for every hop, so a redirect cannot lead into the private network either.
        await asyncio.to_thread(check_public_url, str(request.url), self.settings.allow_private_urls, self.resolve)

    async def aclose(self) -> None:
        await self.http.aclose()

    def schemas(self) -> list[dict]:
        tools = [
            _function("web_search", "Search the web (DuckDuckGo). Returns titles, URLs and snippets.",
                      {"query": {"type": "string"}, "max_results": {"type": "integer", "description": "1-10, default 5"}},
                      ["query"]),
            _function("open_url", "Open a web page and return its main text plus the image URLs found on it.",
                      {"url": {"type": "string", "description": "Full http(s) URL"}}, ["url"]),
            _function("list_files", "List files and folders in the workspace (or a subfolder of it).",
                      {"path": {"type": "string", "description": "Subfolder, default '.'"}}),
            _function("read_file", "Read a text file from the workspace.",
                      {"path": {"type": "string", "description": "Relative path, e.g. notes/todo.md"}}, ["path"]),
            _function("write_file", "Create or overwrite a text file in the workspace; folders are created automatically.",
                      {"path": {"type": "string"}, "content": {"type": "string"},
                       "append": {"type": "boolean", "description": "Append instead of overwrite"}}, ["path", "content"]),
        ]
        if self.settings.vision:
            tools.insert(2, _function("view_image", "Download an image by URL and look at it.",
                                      {"url": {"type": "string", "description": "Direct image URL"}}, ["url"]))
        return tools + (self.mcp.tools if self.mcp else [])

    async def call(self, name: str, args: dict) -> ToolResult:
        if name == "web_search":
            return await self.web_search(str(args.get("query", "")), args.get("max_results", 5)), None
        if name == "open_url":
            return await self.open_url(str(args["url"]))
        if name == "view_image" and self.settings.vision:
            return await self.view_image(str(args["url"]))
        if name == "list_files":
            return self.list_files(str(args.get("path", "."))), None
        if name == "read_file":
            return self.read_file(str(args["path"])), None
        if name == "write_file":
            return self.write_file(str(args["path"]), str(args.get("content", "")), flag(args.get("append"))), None
        if self.mcp and name in self.mcp.routes:
            text, image = await self.mcp.call(name, args)
            return text, image if self.settings.vision else None
        return f"Unknown tool: {name}", None

    async def web_search(self, query: str, max_results: Any = 5) -> str:
        limit = max(1, min(int(max_results or 5), 10))
        results = await asyncio.to_thread(self.search, query, limit)
        if not results:
            return "No results."
        return "\n".join(f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}"
                         for i, r in enumerate(results, 1))

    async def _download(self, url: str) -> tuple[str, str, bytes, str]:
        async with self.http.stream("GET", url, follow_redirects=True, timeout=25) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self.settings.max_download_bytes:
                    raise ValueError(f"Response is larger than {self.settings.max_download_bytes} bytes")
            return str(response.url), response.headers.get("content-type", ""), bytes(body), response.encoding or "utf-8"

    async def open_url(self, url: str) -> ToolResult:
        final_url, content_type, body, encoding = await self._download(url)
        if content_type.startswith("image/"):
            if not self.settings.vision:
                return "This URL is an image, and the current model cannot look at images.", None
            return "This URL is an image; it is attached for you to look at.", image_to_data_url(body, self.settings.image_max_side)
        html = body.decode(encoding, errors="replace")
        text = html_to_text(html)
        title = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.S | re.I)
        limit = self.settings.max_page_chars
        out = f"URL: {final_url}\nTitle: {unescape(title.group(1).strip()) if title else ''}\n\n{text[:limit]}"
        if len(text) > limit:
            out += "\n\n[...text truncated...]"
        images = extract_images(html, final_url)
        if images and self.settings.vision:
            out += "\n\nImages on the page (use view_image to look):\n" + "\n".join(images)
        return out, None

    async def view_image(self, url: str) -> ToolResult:
        _, content_type, body, _ = await self._download(url)
        if "svg" in content_type or url.lower().split("?")[0].endswith(".svg"):
            return "This is an SVG (vector) file and cannot be viewed as a picture.", None
        if content_type and not content_type.startswith("image/"):
            return f"The URL is not an image (content-type: {content_type}); use open_url to read it.", None
        return f"The image from {url} is attached below.", image_to_data_url(body, self.settings.image_max_side)

    def list_files(self, path: str = ".") -> str:
        root = workspace_path(self.settings.workspace, path)
        if not root.is_dir():
            return f"Not a folder: {path}"
        lines = []
        base = self.settings.workspace.resolve()
        for item in sorted(root.rglob("*"))[:200]:
            relative = item.relative_to(base)
            lines.append(f"{relative}/" if item.is_dir() else f"{relative}  ({item.stat().st_size} B)")
        return "\n".join(lines) or "(empty)"

    def read_file(self, path: str) -> str:
        file = workspace_path(self.settings.workspace, path)
        if not file.is_file():
            return f"No such file: {path}"
        text = file.read_text(errors="replace")
        limit = self.settings.max_page_chars
        return text if len(text) <= limit else text[:limit] + "\n[...truncated...]"

    def write_file(self, path: str, content: str, append: bool = False) -> str:
        file = workspace_path(self.settings.workspace, path)
        if file == self.settings.workspace.resolve():
            return "Error: give a file name"
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(content)
        relative = file.relative_to(self.settings.workspace.resolve()).as_posix()
        return f"Saved {relative} ({file.stat().st_size} B). Link for the user: /files/{relative}"
