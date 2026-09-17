"""Central configuration. Values come from the environment (.env) with safe defaults.

Every autonomy threshold lives here so the "how much do we trust the agent"
policy is in one auditable place, not scattered through the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
BRAND = "Spotify"


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    # Deterministic-ish generation: we want reproducible, conservative behaviour.
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    max_retries: int = 4
    timeout_s: int = 60


@dataclass(frozen=True)
class AutonomyThresholds:
    """The gates that decide auto-handle vs. escalate. Tunable, and every one is
    logged in the decision trace so a reviewer can see *why* a message was gated."""

    intent_confidence: float = 0.60   # classifier must be at least this sure
    retrieval_similarity: float = 0.18  # best grounding evidence must be this close
    groundedness: float = 0.66         # verifier must judge the draft this grounded
    # Intents we never auto-send even when confident: money and account security.
    sensitive_intents: tuple[str, ...] = ("billing_payment", "account_access")


@dataclass(frozen=True)
class Settings:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    thresholds: AutonomyThresholds = field(default_factory=AutonomyThresholds)
    corpus_csv: Path = DATA_DIR / "raw" / "spotify_threads.csv"
    golden_path: Path = DATA_DIR / "golden" / "golden_test.jsonl"
    train_path: Path = DATA_DIR / "golden" / "train_labeled.jsonl"
    cache_dir: Path = ROOT / ".cache"


SETTINGS = Settings()
