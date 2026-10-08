"""The agent loop: ask the model, run the tools it calls, feed results back, stream every step."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from .config import Settings
from .tools import Toolbox

SYSTEM_PROMPT = """You are a direct, helpful assistant running locally with tools.
Tools:
- web_search: find fresh information on the web.
- open_url: open a link and read its text (it also lists image links on the page).
- view_image: look at an image by URL (photo, screenshot, chart), when available.
- list_files / read_file / write_file: work with files in the user's workspace folder (relative paths).
- Tools named like `server__tool` come from connected MCP servers; use them when they fit the task.
Rules:
- Act, do not announce: when a step needs a tool, call the tool now instead of saying you will.
- Never claim you saved, ran, checked or opened something unless a tool result in this conversation shows it.
- When the user asks to create, save or export something (code, notes, report, CSV, HTML), write it with
  write_file and give the user the file link. Do not paste the whole file back into the chat.
- If the user gives a link, open it with open_url before answering. Do not answer from memory about a page you have not opened.
- For anything current (news, prices, versions, people's roles) use web_search, then open_url on the best 1-3 results.
- If you used web pages, list their URLs at the end; if you did not, do not mention URLs.
- Reply in the language of the user's latest message and never switch languages mid-answer. Be concise."""


class LLMError(RuntimeError):
    pass


class LLM:
    """Any OpenAI-compatible chat completions endpoint: llama.cpp, Ollama, LM Studio, vLLM or a hosted API."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        self.http = httpx.AsyncClient(base_url=settings.llm_base_url, headers=headers, timeout=900, transport=transport)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {"model": self.settings.llm_model, "messages": messages,
                   "temperature": self.settings.llm_temperature, **self.settings.llm_extra}
        if tools:
            payload |= {"tools": tools, "tool_choice": "auto"}
        response = await self.http.post("/chat/completions", json=payload)
        if response.status_code != 200:
            raise LLMError(f"LLM {response.status_code}: {response.text[:500]}")
        return response.json()["choices"][0]["message"]

    async def healthy(self) -> bool:
        try:
            return (await self.http.get("/models", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self.http.aclose()


def without_images(history: list[dict]) -> list[dict]:
    """For text-only models: keep the words of messages that carried pictures."""
    cleaned = []
    for message in history:
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
            message = {**message, "content": text + "\n[An image was attached, but this model cannot see images.]"}
        cleaned.append(message)
    return cleaned


async def run_agent(history: list[dict], llm: LLM, toolbox: Toolbox) -> AsyncIterator[dict]:
    """Yields {type: tool|image|answer|error} events until the model answers or the step limit is reached."""
    settings = toolbox.settings
    if not settings.vision:
        history = without_images(history)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    try:
        for _ in range(settings.max_steps):
            message = await llm.chat(messages, toolbox.schemas())
            calls = message.get("tool_calls") or []
            if not calls:
                yield {"type": "answer", "content": message.get("content") or ""}
                return
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
            images = []
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool", "name": name, "args": args}
                try:
                    result, image = await toolbox.call(name, args)
                except Exception as error:  # noqa: BLE001 — tool failures go back to the model as text
                    result, image = f"Error: {type(error).__name__}: {error}", None
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": result})
                if image:
                    images.append(image)
                    yield {"type": "image", "src": image}
            if images:
                # Pictures go in a separate user turn: that is the form multimodal chat templates understand.
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Here are the images you requested:"},
                    *({"type": "image_url", "image_url": {"url": src}} for src in images),
                ]})
        messages.append({"role": "user", "content": "Answer now using what you have gathered."})
        final = await llm.chat(messages)
        yield {"type": "answer", "content": final.get("content") or ""}
    except (LLMError, httpx.HTTPError) as error:
        yield {"type": "error", "content": str(error) or type(error).__name__}
