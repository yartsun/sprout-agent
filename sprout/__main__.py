"""`python -m sprout` or `sprout`: start the agent on http://127.0.0.1:3300."""
import uvicorn

from .app import create_app
from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    print(f"Sprout on http://{settings.host}:{settings.port} · model {settings.llm_model} at {settings.llm_base_url}")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
