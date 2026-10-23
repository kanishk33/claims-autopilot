"""Settings, file paths, and logging.

Everything configurable comes from environment variables (optionally loaded
from a `.env` file). Secrets are never hardcoded and never logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repository root: src/claims_autopilot/config.py -> up three levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RULES_PATH = DATA_DIR / "validation_rules.yml"
REFERENCE_PATH = DATA_DIR / "carc_rarc_reference.csv"
KNOWLEDGE_PATH = DATA_DIR / "denial_knowledge_base.md"
SYNTHETIC_CLAIMS_DIR = DATA_DIR / "synthetic_claims"
SYNTHETIC_DENIALS_DIR = DATA_DIR / "synthetic_denials"

DEFAULT_MODEL = "gpt-4.1-mini"
PLACEHOLDER_KEYS = {"", "sk-your-key-here", "your-api-key", "changeme"}

load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    timeout_seconds: float
    log_level: str
    db_path: Path

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key) and self.openai_api_key not in PLACEHOLDER_KEYS

    def masked_key(self) -> str:
        """Show only the last 4 characters, e.g. 'sk-...abcd'."""
        if not self.has_api_key:
            return "(not set)"
        return f"{self.openai_api_key[:3]}...{self.openai_api_key[-4:]}"


def get_settings() -> Settings:
    """Read settings fresh from the environment each time (cheap, and lets the
    Streamlit app pick up changes without a restart)."""
    db = Path(os.getenv("CLAIMS_DB_PATH", "outputs/claims_autopilot.db"))
    if not db.is_absolute():
        db = PROJECT_ROOT / db
    return Settings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        openai_model=(os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip(),
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        db_path=db,
    )


def config_problems(settings: Settings | None = None) -> list[str]:
    """Plain-language problems with the configuration (empty list = looks OK).

    This only checks the *configuration*. A well-formed key can still fail for
    quota, billing, or model-access reasons; use `llm.check_api_access()`.
    """
    s = settings or get_settings()
    problems: list[str] = []
    if not s.openai_api_key:
        problems.append("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    elif s.openai_api_key in PLACEHOLDER_KEYS:
        problems.append("OPENAI_API_KEY still has the placeholder value from .env.example.")
    elif not s.openai_api_key.startswith("sk-"):
        problems.append("OPENAI_API_KEY does not start with 'sk-'. Check that you copied the whole key.")
    if not s.openai_model:
        problems.append("OPENAI_MODEL is empty.")
    return problems


def setup_logging() -> None:
    level = getattr(logging, get_settings().log_level, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The OpenAI/httpx loggers can print request details at DEBUG; keep them quiet.
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
