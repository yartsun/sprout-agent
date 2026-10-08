"""Connects the MCP servers listed in mcp.json (the Claude Desktop / Cursor format).

Each server's tools reach the model as `<server>__<tool>`. A server that fails to
start is reported in /api/tools and skipped; the agent keeps working without it.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .tools import image_to_data_url


class MCPHub:
    def __init__(self, config: Path, timeout: float = 60, max_chars: int = 8000):
        self.config = Path(config)
        self.timeout = timeout
        self.max_chars = max_chars
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.routes: dict[str, tuple[str, str]] = {}  # model-facing name -> (server, tool)
        self.tools: list[dict] = []                    # OpenAI function schemas
        self.errors: dict[str, str] = {}

    async def start(self) -> None:
        if not self.config.exists():
            return
        servers = json.loads(self.config.read_text(encoding="utf-8")).get("mcpServers", {})
        for name, cfg in servers.items():
            if cfg.get("disabled"):
                continue
            # anyio requires entering and leaving these contexts in the same task, so no wait_for here.
            server_stack = AsyncExitStack()
            try:
                await self._connect(name, cfg, server_stack)
                self.stack.push_async_callback(server_stack.aclose)
            except Exception as error:  # noqa: BLE001 — one broken server must not stop the others
                self.errors[name] = f"{type(error).__name__}: {error}"
                try:
                    await server_stack.aclose()
                except Exception:  # noqa: BLE001
                    pass

    async def _connect(self, name: str, cfg: dict, stack: AsyncExitStack) -> None:
        if "url" in cfg:
            if cfg.get("transport") == "sse" or cfg["url"].rstrip("/").endswith("/sse"):
                read, write = await stack.enter_async_context(sse_client(cfg["url"], headers=cfg.get("headers")))
            else:
                read, write, _ = await stack.enter_async_context(streamablehttp_client(cfg["url"], headers=cfg.get("headers")))
        else:
            # "python" means the agent's own interpreter, so servers share its virtual environment.
            command = sys.executable if cfg["command"] in ("python", "python3") else cfg["command"]
            params = StdioServerParameters(command=command, args=cfg.get("args", []),
                                           env={**os.environ, **cfg.get("env", {})},
                                           cwd=cfg.get("cwd", str(self.config.resolve().parent)))
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = session
        for tool in (await session.list_tools()).tools:
            alias = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{name}__{tool.name}")[:64]
            self.routes[alias] = (name, tool.name)
            self.tools.append({"type": "function", "function": {
                "name": alias,
                "description": f"[MCP {name}] {tool.description or tool.name}"[:1024],
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            }})

    async def call(self, alias: str, args: dict) -> tuple[str, str | None]:
        server, tool = self.routes[alias]
        result = await asyncio.wait_for(self.sessions[server].call_tool(tool, args), timeout=self.timeout)
        texts, image = [], None
        for item in result.content:
            if item.type == "text":
                texts.append(item.text)
            elif item.type == "image" and image is None:
                image = image_to_data_url(base64.b64decode(item.data))
            elif item.type == "resource":
                texts.append(getattr(item.resource, "text", "") or f"[resource {item.resource.uri}]")
        text = "\n".join(texts) or ("(no output)" if image is None else "Image attached below.")
        if result.isError:
            text = "Error: " + text
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[...truncated...]"
        return text, image

    async def stop(self) -> None:
        await self.stack.aclose()
