"""Pipeline configuration: config.yaml, with RAG_<KEY> environment overrides."""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and apply RAG_<KEY> env overrides (values parsed as YAML)."""
    cfg_path = Path(path or os.getenv("RAG_CONFIG", ROOT / "config.yaml"))
    cfg: dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    unknown = [k for k in os.environ if k.startswith("RAG_") and k != "RAG_CONFIG" and k[4:].lower() not in cfg]
    if unknown:  # a typo'd override would otherwise be silently ignored
        raise ValueError(f"Unknown config override(s): {unknown}. Valid keys: {sorted(cfg)}")
    for key in cfg:
        env = os.getenv(f"RAG_{key.upper()}")
        if env is not None:
            cfg[key] = yaml.safe_load(env)
    return cfg


def path(key: str) -> Path:
    """Resolve a path-valued config key against the repo root (absolute values pass through)."""
    return ROOT / CFG[key]


CFG: dict = load_config()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger("compliance-copilot")
