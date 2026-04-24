"""Centralised API-key loader. Reads .env in repo root.

Usage:
  from _meis_keys import GPT_KEY, GEMINI_KEY, RUOLI_BASE_URL

If a required key is missing, raises immediately with a helpful
message rather than letting downstream API calls fail mysteriously.
"""

from __future__ import annotations

import os
from pathlib import Path

# Try python-dotenv first; fall back to a tiny parser
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required env var {name!r} not set. "
            f"Add it to MEIS/.env (which is .gitignore'd) or export "
            f"it before running."
        )
    return val


GPT_KEY = _require("OPENAI_API_KEY")
GEMINI_KEY = _require("GEMINI_API_KEY")
RUOLI_BASE_URL = os.environ.get("RUOLI_BASE_URL",
                                  os.environ.get("OPENAI_BASE_URL",
                                                 "https://ruoli.dev/v1"))
