import asyncio
import json
from pathlib import Path

from sprout.mcp_hub import MCPHub

SERVER = str(Path(__file__).with_name("fake_mcp_server.py"))


def test_connects_stdio_servers_and_routes_their_tools(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {
        "fake": {"command": "python", "args": [SERVER]},
        "off": {"command": "python", "args": [SERVER], "disabled": True},
        "broken": {"command": "definitely-not-a-command-xyz"},
    }}))

    async def scenario():
        hub = MCPHub(config, timeout=20)
        await hub.start()
        try:
            assert set(hub.routes) == {"fake__shout", "fake__swatch", "fake__fail"}
            assert "broken" in hub.errors and "off" not in hub.errors
            assert hub.tools[0]["function"]["description"].startswith("[MCP fake]")
            assert await hub.call("fake__shout", {"text": "hi"}) == ("HI", None)
            text, image = await hub.call("fake__swatch", {})
            assert image.startswith("data:image/jpeg;base64,")
            text, _ = await hub.call("fake__fail", {})
            assert text.startswith("Error:") and "broken on purpose" in text
        finally:
            await hub.stop()

    asyncio.run(scenario())


def test_missing_config_means_no_servers(tmp_path):
    hub = MCPHub(tmp_path / "nope.json")
    asyncio.run(hub.start())
    assert hub.tools == [] and hub.errors == {}
