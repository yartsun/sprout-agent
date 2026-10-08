"""MCP server with one tool, `run_code`: Python, JavaScript or bash in a throwaway Docker container.

The container has no network, 512 MB of memory, one CPU, no Linux capabilities and a read-only
root file system. Only the workspace folder is mounted, so the model can run code against files it wrote.
Add it to mcp.json: {"sandbox": {"command": "python", "args": ["-m", "sprout.sandbox_server"]}}
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKSPACE = Path(os.getenv("WORKSPACE", "workspace")).resolve()
IMAGE = os.getenv("SANDBOX_IMAGE", "sprout-sandbox")
MAX_OUTPUT = 8000
RUNNERS = {
    "python": ("main.py", ["python", "/code/main.py"]),
    "javascript": ("main.js", ["node", "/code/main.js"]),
    "bash": ("main.sh", ["bash", "/code/main.sh"]),
}
ALIASES = {"py": "python", "js": "javascript", "node": "javascript", "sh": "bash"}

server = FastMCP("sandbox", log_level="WARNING")


def _cut(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"\n[...truncated, {len(text)} chars total...]"


def docker_command(name: str, code_dir: str, language: str, timeout: int) -> list[str]:
    _, command = RUNNERS[language]
    # Run as the host user: the 0700 temp dir stays readable and workspace files stay owned by you (Linux).
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    return [
        "docker", "run", "--rm", "-i", "--name", name, *user,
        "--network", "none", "--memory", "512m", "--cpus", "1", "--pids-limit", "256",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m", "-e", "HOME=/tmp",
        "-v", f"{code_dir}:/code:ro", "-v", f"{WORKSPACE}:/workspace", "-w", "/workspace",
        IMAGE, "timeout", "-s", "KILL", str(timeout), *command,
    ]


def sandbox_run(language: str, code: str, stdin: str = "", timeout: int = 30) -> dict:
    language = ALIASES.get(language, language)
    if language not in RUNNERS:
        return {"exit_code": -1, "stdout": "", "stderr": f"Unsupported language: {language}", "timed_out": False}
    timeout = max(1, min(int(timeout or 30), 120))
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    name = f"sprout-run-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory() as code_dir:
        Path(code_dir, RUNNERS[language][0]).write_text(code, encoding="utf-8")
        try:
            done = subprocess.run(docker_command(name, code_dir, language, timeout), input=stdin, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return {"exit_code": -1, "stdout": "", "stderr": "", "timed_out": True}
        except FileNotFoundError:
            return {"exit_code": -1, "stdout": "", "stderr": "Docker is not installed or not running.", "timed_out": False}
    # 137 = killed by SIGKILL from `timeout` (or the memory limit) inside the container.
    return {"exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr, "timed_out": done.returncode == 137}


@server.tool()
def run_code(language: str, code: str, stdin: str = "", timeout: int = 30) -> str:
    """Run code in an isolated Docker sandbox and return the exit code, stdout and stderr.
    language: python | javascript | bash. No network, 512 MB RAM, 1 CPU, timeout up to 120 s.
    Python has numpy, pandas, matplotlib, sympy, pytest and beautifulsoup4; Node.js has its built-ins.
    The working directory /workspace is the user's workspace folder, so files from write_file are there.
    Use it to check code before giving it to the user: run a few cases, read the errors, fix, run again."""
    result = sandbox_run(language, code, stdin, timeout)
    if result["timed_out"]:
        return f"KILLED: timeout ({timeout}s) or memory limit (512 MB)\nstdout:\n{_cut(result['stdout'])}"
    return f"exit_code: {result['exit_code']}\nstdout:\n{_cut(result['stdout'])}\nstderr:\n{_cut(result['stderr'])}"


if __name__ == "__main__":
    server.run()
