"""Run the test suite against a fresh, disposable seed calendar."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


def main() -> int:
    sandbox_env = "CALENDAR_SANDBOX_DIR"
    log_env = "TOOL_LOG_PATH"
    evaluation_env = "CALENDAR_AGENT_EVALUATION_SANDBOX"
    previous = {
        name: os.environ.get(name)
        for name in (sandbox_env, log_env, evaluation_env)
    }

    try:
        with tempfile.TemporaryDirectory(prefix="calendar-agent-evaluation-") as root:
            os.environ[sandbox_env] = root
            os.environ[log_env] = str(Path(root) / "tool_calls.jsonl")
            os.environ[evaluation_env] = root

            from sandbox.store import reset

            reset()
            project_root = Path(__file__).resolve().parents[1]
            suite = unittest.defaultTestLoader.discover(
                start_dir=str(project_root / "tests"),
                pattern="test*.py",
                top_level_dir=str(project_root),
            )
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    sys.exit(main())
