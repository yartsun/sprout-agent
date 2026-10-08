"""Runs real containers; enabled with SANDBOX_TESTS=1 after `docker build -t sprout-sandbox sandbox`."""
import os

import pytest

from sprout import sandbox_server

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(os.getenv("SANDBOX_TESTS") != "1", reason="set SANDBOX_TESTS=1 to run Docker sandbox tests"),
]


def test_runs_python_and_reads_stdin():
    result = sandbox_server.sandbox_run("py", "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()))", stdin="2 3 4")
    assert result == {"exit_code": 0, "stdout": "9\n", "stderr": "", "timed_out": False}


def test_has_no_network():
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)"
    result = sandbox_server.sandbox_run("python", code)
    assert result["exit_code"] != 0 and "Error" in result["stderr"]


def test_timeout_kills_the_process():
    result = sandbox_server.sandbox_run("bash", "sleep 30", timeout=2)
    assert result["timed_out"] is True


def test_root_file_system_is_read_only_but_workspace_is_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_server, "WORKSPACE", tmp_path)
    result = sandbox_server.sandbox_run("bash", "touch /etc/x 2>/dev/null || echo ro; echo ok > out.txt; cat out.txt")
    assert result["stdout"] == "ro\nok\n"
    assert (tmp_path / "out.txt").read_text() == "ok\n"


def test_unknown_language_is_rejected_without_docker():
    assert sandbox_server.sandbox_run("cobol", "DISPLAY 'HI'")["stderr"] == "Unsupported language: cobol"
