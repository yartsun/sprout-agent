import asyncio
import json
from dataclasses import replace

import httpx
from conftest import FakeLLM, answer, png_bytes, public_resolver, tool_call, web_transport

from sprout.agent import LLM, run_agent
from sprout.tools import Toolbox


def run(settings, llm: FakeLLM, history, pages=None):
    toolbox = Toolbox(settings, transport=web_transport(pages or {}), resolve=public_resolver)

    async def collect():
        return [event async for event in run_agent(history, LLM(settings, transport=llm.transport()), toolbox)]

    return asyncio.run(collect())


def test_runs_tools_until_the_model_answers(settings):
    settings.workspace.mkdir(parents=True)
    llm = FakeLLM([tool_call("write_file", {"path": "plan.md", "content": "# Plan\n"}), answer("Saved to /files/plan.md")])
    events = run(settings, llm, [{"role": "user", "content": "Make a plan file"}])
    assert events == [
        {"type": "tool", "name": "write_file", "args": {"path": "plan.md", "content": "# Plan\n"}},
        {"type": "answer", "content": "Saved to /files/plan.md"},
    ]
    assert (settings.workspace / "plan.md").read_text() == "# Plan\n"
    second = llm.requests[1]["messages"]
    assert second[0]["role"] == "system"
    assert second[-1]["role"] == "tool" and "Saved plan.md" in second[-1]["content"]
    assert llm.requests[0]["model"] == "llama3.1" and llm.requests[0]["tool_choice"] == "auto"


def test_tool_errors_go_back_to_the_model(settings):
    llm = FakeLLM([tool_call("read_file", {"path": "../../etc/passwd"}), answer("I cannot read that.")])
    events = run(settings, llm, [{"role": "user", "content": "read it"}])
    assert events[-1] == {"type": "answer", "content": "I cannot read that."}
    assert "Error: ValueError: Path must stay inside the workspace folder" in llm.requests[1]["messages"][-1]["content"]


def test_images_are_forwarded_in_a_user_turn(settings):
    pages = {"https://example.com/cat.png": httpx.Response(200, headers={"content-type": "image/png"}, content=png_bytes())}
    llm = FakeLLM([tool_call("view_image", {"url": "https://example.com/cat.png"}), answer("A green square.")])
    events = run(settings, llm, [{"role": "user", "content": "what is it?"}], pages)
    assert [event["type"] for event in events] == ["tool", "image", "answer"]
    last = llm.requests[1]["messages"][-1]
    assert last["role"] == "user" and last["content"][1]["image_url"]["url"].startswith("data:image/jpeg")


def test_text_only_models_never_receive_images(settings):
    text_only = replace(settings, vision=False)
    history = [{"role": "user", "content": [{"type": "text", "text": "Describe"},
                                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}]}]
    llm = FakeLLM([answer("I cannot see images.")])
    run(text_only, llm, history)
    sent = llm.requests[0]["messages"][1]["content"]
    assert isinstance(sent, str) and sent.startswith("Describe") and "cannot see images" in sent


def test_step_limit_forces_a_final_answer(settings):
    limited = replace(settings, max_steps=2)
    llm = FakeLLM([tool_call("list_files", {}), tool_call("list_files", {}), answer("Here is what I found.")])
    events = run(limited, llm, [{"role": "user", "content": "loop"}])
    assert events[-1] == {"type": "answer", "content": "Here is what I found."}
    assert "tools" not in llm.requests[-1]  # the final call cannot start more tool calls
    assert llm.requests[-1]["messages"][-1]["content"] == "Answer now using what you have gathered."


def test_model_errors_become_an_error_event(settings):
    events = run(settings, FakeLLM([], status=500), [{"role": "user", "content": "hi"}])
    assert events == [{"type": "error", "content": "LLM 500: model exploded"}]


def test_api_key_and_extra_sampling_options_are_sent(settings):
    keyed = replace(settings, llm_api_key="secret-key", llm_extra={"top_k": 20})
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"choices": [{"message": answer("ok")}]})

    async def go():
        return await LLM(keyed, transport=httpx.MockTransport(handler)).chat([{"role": "user", "content": "hi"}])

    assert asyncio.run(go())["content"] == "ok"
    assert seen["auth"] == "Bearer secret-key"
    assert json.loads(seen["body"])["top_k"] == 20
