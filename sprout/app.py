"""HTTP layer: the chat UI, a streaming chat endpoint and the workspace file server."""
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .agent import LLM, run_agent
from .config import Settings
from .mcp_hub import MCPHub
from .tools import Toolbox, workspace_path

STATIC = Path(__file__).parent / "static"
MAX_MESSAGES = 200


def create_app(
    settings: Settings | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
    web_transport: httpx.AsyncBaseTransport | None = None,
    search: Callable | None = None,
    resolve: Callable | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    hub = MCPHub(settings.mcp_config, settings.mcp_timeout, settings.max_page_chars)
    extra = {key: value for key, value in {"search": search, "resolve": resolve}.items() if value}
    toolbox = Toolbox(settings, hub, transport=web_transport, **extra)
    llm = LLM(settings, transport=llm_transport)
    hosts = {f"127.0.0.1:{settings.port}", f"localhost:{settings.port}", *settings.allowed_hosts}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.workspace.mkdir(parents=True, exist_ok=True)
        await hub.start()
        yield
        await hub.stop()
        await toolbox.aclose()
        await llm.aclose()

    app = FastAPI(title="Sprout", lifespan=lifespan)

    def local_only(request: Request) -> None:
        """The agent can write files and run code, so only this machine's own pages may drive it.

        Host blocks DNS rebinding; Origin blocks other sites in the same browser.
        """
        host = request.headers.get("host", "")
        if host not in hosts:
            raise HTTPException(403, "Unexpected Host header")
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://{h}" for h in hosts}:
            raise HTTPException(403, "Cross-origin requests are not allowed")

    @app.get("/")
    async def index(request: Request):
        local_only(request)
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health(request: Request):
        local_only(request)
        return {"llm": await llm.healthy(), "model": settings.llm_model}

    @app.get("/api/tools")
    async def tools(request: Request):
        local_only(request)
        return {
            "builtin": [tool["function"]["name"] for tool in toolbox.schemas() if "__" not in tool["function"]["name"]],
            "mcp": list(hub.routes),
            "mcp_errors": hub.errors,
            "vision": settings.vision,
            "model": settings.llm_model,
        }

    @app.get("/files/{path:path}")
    async def files(path: str, request: Request):
        local_only(request)
        try:
            file = workspace_path(settings.workspace, path)
        except ValueError:
            raise HTTPException(403) from None
        if not file.is_file():
            raise HTTPException(404)
        # Model-written HTML runs in an opaque origin: it cannot call this API or read other files.
        return FileResponse(file, headers={
            "Content-Security-Policy": "sandbox allow-scripts allow-popups allow-forms",
            "X-Content-Type-Options": "nosniff",
        })

    @app.post("/api/chat")
    async def chat(request: Request):
        local_only(request)
        # A JSON content type forces a CORS preflight, so a plain form or text/plain POST cannot start the agent.
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "Use application/json")
        body = await request.json()
        history = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(history, list) or not history or len(history) > MAX_MESSAGES:
            raise HTTPException(422, f"messages must be a list of 1-{MAX_MESSAGES} chat messages")
        if any(not isinstance(m, dict) or m.get("role") not in ("user", "assistant") for m in history):
            raise HTTPException(422, "Only user and assistant messages are accepted")

        async def stream():
            async for event in run_agent(history, llm, toolbox):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield 'data: {"type": "done"}\n\n'

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    return app
