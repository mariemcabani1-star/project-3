"""Sandboxed calendar backend.

A JSON file stands in for a real calendar API. Nothing here ever touches a real
account: `reset()` restores the seed data, so every eval run starts identical.

Override the location with CALENDAR_SANDBOX_DIR (useful for tests / parallel runs).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED_PATH = HERE / "seed.json"


def sandbox_dir() -> Path:
    return Path(os.environ.get("CALENDAR_SANDBOX_DIR", HERE))


def state_path() -> Path:
    return sandbox_dir() / "state.json"


def reset() -> None:
    """Restore the sandbox to the seed state."""
    sandbox_dir().mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SEED_PATH, state_path())


def load() -> dict:
    if not state_path().exists():
        reset()
    with open(state_path(), encoding="utf-8") as f:
        return json.load(f)


def save(state: dict) -> None:
    tmp = state_path().with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path())  # atomic: no half-written state
