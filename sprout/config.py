"""Settings from environment variables, read once at startup."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


def _bool(value: str | None, default: bool) -> bool:
    return default if value is None or value == "" else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "llama3.1"
    llm_api_key: str = ""
    llm_temperature: float = 0.3
    llm_extra: dict = field(default_factory=dict)
    vision: bool = True
    max_steps: int = 12
    workspace: Path = Path("workspace")
    max_page_chars: int = 8000
    max_download_bytes: int = 10_000_000
    image_max_side: int = 1024
    mcp_config: Path = Path("mcp.json")
    mcp_timeout: float = 60
    allow_private_urls: bool = False
    host: str = "127.0.0.1"
    port: int = 3300
    allowed_hosts: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        extra = env.get("LLM_EXTRA_BODY", "")
        port = int(env.get("PORT", cls.port))
        return cls(
            llm_base_url=env.get("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            llm_api_key=env.get("LLM_API_KEY", ""),
            llm_temperature=float(env.get("LLM_TEMPERATURE", cls.llm_temperature)),
            llm_extra=json.loads(extra) if extra else {},
            vision=_bool(env.get("LLM_VISION"), True),
            max_steps=int(env.get("MAX_STEPS", cls.max_steps)),
            workspace=Path(env.get("WORKSPACE", "workspace")).resolve(),
            max_page_chars=int(env.get("MAX_PAGE_CHARS", cls.max_page_chars)),
            max_download_bytes=int(env.get("MAX_DOWNLOAD_BYTES", cls.max_download_bytes)),
            image_max_side=int(env.get("IMAGE_MAX_SIDE", cls.image_max_side)),
            mcp_config=Path(env.get("MCP_CONFIG", "mcp.json")),
            mcp_timeout=float(env.get("MCP_TIMEOUT", cls.mcp_timeout)),
            allow_private_urls=_bool(env.get("ALLOW_PRIVATE_URLS"), False),
            host=env.get("HOST", cls.host),
            port=port,
            allowed_hosts=tuple(h.strip() for h in env.get("ALLOWED_HOSTS", "").split(",") if h.strip()),
        )
