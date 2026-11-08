"""Pipeline configuration: config.yaml, with RAG_<KEY> environment overrides."""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


PRESET: str = os.getenv("RAG_PRESET", "default")


def load_config(path: str | Path | None = None, preset: str = PRESET) -> dict:
    """config.yaml, then presets/<preset>.yaml on top, then RAG_<KEY> env overrides.

    Precedence (last wins): config.yaml < preset < environment.
    """
    cfg_path = Path(path or os.getenv("RAG_CONFIG", ROOT / "config.yaml"))
    cfg: dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    preset_path = ROOT / "presets" / f"{preset}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Unknown preset {preset!r}: {preset_path} does not exist")
    overlay = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    bad = sorted(set(overlay) - set(cfg))
    if bad:
        raise ValueError(f"Preset {preset!r} sets unknown key(s) {bad}")
    cfg.update(overlay)
    unknown = [k for k in os.environ if k.startswith("RAG_") and k not in ("RAG_CONFIG", "RAG_PRESET") and k[4:].lower() not in cfg]
    if unknown:  # a typo'd override would otherwise be silently ignored
        raise ValueError(f"Unknown config override(s): {unknown}. Valid keys: {sorted(cfg)}")
    for key in cfg:
        env = os.getenv(f"RAG_{key.upper()}")
        if env is not None:  # string keys stay raw: YAML would turn "search_query: " into a dict
            cfg[key] = env if isinstance(cfg[key], str) else yaml.safe_load(env)
    return cfg


def path(key: str) -> Path:
    """Resolve a path-valued config key against the repo root (absolute values pass through)."""
    return ROOT / CFG[key]


def chunks_file(domain: str) -> Path:
    """Chunk files are per collection, so presets with different chunking don't overwrite each other."""
    return path("processed_dir") / "chunks" / CFG["collection"] / f"{domain}.json"


CFG: dict = load_config()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger("compliance-copilot")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)  # posthog version-mismatch noise; telemetry is off
