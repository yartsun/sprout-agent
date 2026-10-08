import asyncio
import base64
import io

import httpx
import pytest
from conftest import png_bytes, public_resolver, web_transport
from PIL import Image

from sprout.tools import BlockedURL, Toolbox, check_public_url, extract_images, image_to_data_url, workspace_path


def make(settings, pages=None, **kwargs):
    return Toolbox(settings, transport=web_transport(pages or {}), resolve=public_resolver, **kwargs)


def test_workspace_paths_cannot_escape(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    assert workspace_path(root, "notes/a.md") == (root / "notes/a.md").resolve()
    assert workspace_path(root, "/etc/passwd") == (root / "etc/passwd").resolve()  # absolute paths stay inside
    with pytest.raises(ValueError):
        workspace_path(root, "../outside.txt")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        workspace_path(root, "link/secret.txt")


def test_files_round_trip(settings):
    tools = make(settings)
    settings.workspace.mkdir(parents=True)
    assert "Link for the user: /files/reports/q3.md" in tools.write_file("reports/q3.md", "# Q3\n")
    tools.write_file("reports/q3.md", "more\n", append=True)
    assert tools.read_file("reports/q3.md") == "# Q3\nmore\n"
    assert "reports/q3.md" in tools.list_files()
    assert tools.write_file(".", "x") == "Error: give a file name"
    assert tools.read_file("missing.txt") == "No such file: missing.txt"


@pytest.mark.parametrize("url", [
    "http://localhost:8080/admin", "http://127.0.0.1/", "http://10.0.0.5/", "http://192.168.1.1/",
    "http://metadata.internal/latest/meta-data", "http://[::1]/", "file:///etc/passwd", "ftp://example.com/",
])
def test_private_and_non_http_urls_are_blocked(url):
    def resolve(host, port, *args, **kwargs):
        address = host.strip("[]") if host[0].isdigit() or ":" in host else public_resolver(host, port)[0][4][0]
        return [(2, 1, 6, "", (address, 0))]

    with pytest.raises(BlockedURL):
        check_public_url(url, resolve=resolve)


def test_public_urls_pass_and_private_ones_can_be_allowed():
    check_public_url("https://example.com/page", resolve=public_resolver)
    check_public_url("http://localhost:8080/", allow_private=True, resolve=public_resolver)


def test_redirect_into_the_private_network_is_blocked(settings):
    pages = {"https://example.com/go": httpx.Response(302, headers={"location": "http://localhost:8080/admin"})}
    with pytest.raises(BlockedURL):
        asyncio.run(make(settings, pages).open_url("https://example.com/go"))


def test_open_url_extracts_text_title_and_images(settings):
    html = (
        "<html><head><title>Launch notes</title><meta property='og:image' content='/cover.png'></head>"
        "<body><article><h1>Launch</h1><p>" + "Sprout reads pages. " * 30 + "</p>"
        "<img src='https://cdn.example.com/a.jpg'><img src='/logo.svg'></article></body></html>"
    )
    pages = {"https://example.com/post": httpx.Response(200, headers={"content-type": "text/html"}, text=html)}
    text, image = asyncio.run(make(settings, pages).open_url("https://example.com/post"))
    assert image is None
    assert "Title: Launch notes" in text
    assert "Sprout reads pages." in text
    assert "[...text truncated...]" in text  # max_page_chars = 200 in the test settings
    assert "https://example.com/cover.png" in text and "https://cdn.example.com/a.jpg" in text
    assert "logo.svg" not in text


def test_downloads_are_capped(settings):
    from dataclasses import replace

    small = replace(settings, max_download_bytes=1000)
    pages = {"https://example.com/big": httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 5000)}
    with pytest.raises(ValueError, match="larger than 1000 bytes"):
        asyncio.run(make(small, pages).open_url("https://example.com/big"))


def test_view_image_resizes_and_rejects_non_images(settings):
    pages = {
        "https://example.com/p.png": httpx.Response(200, headers={"content-type": "image/png"}, content=png_bytes()),
        "https://example.com/page": httpx.Response(200, headers={"content-type": "text/html"}, text="<p>hi</p>"),
    }
    tools = make(settings, pages)
    text, data_url = asyncio.run(tools.view_image("https://example.com/p.png"))
    assert data_url.startswith("data:image/jpeg;base64,")
    assert Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).size == (1024, 512)
    text, data_url = asyncio.run(tools.view_image("https://example.com/page"))
    assert data_url is None and "not an image" in text


def test_text_only_models_get_no_vision_tool(settings):
    from dataclasses import replace

    names = [tool["function"]["name"] for tool in make(replace(settings, vision=False)).schemas()]
    assert "view_image" not in names and "open_url" in names
    assert "view_image" in [tool["function"]["name"] for tool in make(settings).schemas()]


def test_web_search_formats_results(settings):
    def search(query, limit):
        assert (query, limit) == ("sprout agent", 10)
        return [{"title": "Sprout", "href": "https://example.com", "body": "A small agent"}]

    text = asyncio.run(make(settings, search=search).web_search("sprout agent", 50))
    assert text == "1. Sprout\n   https://example.com\n   A small agent"


def test_extract_images_limits_and_deduplicates():
    html = "".join(f"<img src='/i/{n % 5}.png'>" for n in range(40))
    assert extract_images(html, "https://example.com/", limit=3) == [
        "https://example.com/i/0.png", "https://example.com/i/1.png", "https://example.com/i/2.png"]


def test_image_to_data_url_handles_transparency():
    buffer = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(buffer, format="PNG")
    assert image_to_data_url(buffer.getvalue()).startswith("data:image/jpeg;base64,")


def test_string_booleans_from_small_models_are_parsed(settings):
    tools = make(settings)
    settings.workspace.mkdir(parents=True)
    asyncio.run(tools.call("write_file", {"path": "a.txt", "content": "one"}))
    asyncio.run(tools.call("write_file", {"path": "a.txt", "content": "two", "append": "False"}))
    assert tools.read_file("a.txt") == "two"
    asyncio.run(tools.call("write_file", {"path": "a.txt", "content": "+", "append": "true"}))
    assert tools.read_file("a.txt") == "two+"
