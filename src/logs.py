"""Structured JSON logs (one object per line on stderr) carrying the current request id."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_STD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(r.created)) + f".{int(r.msecs):03d}Z",
               "level": r.levelname, "logger": r.name, "request_id": request_id.get(), "msg": r.getMessage()}
        out.update({k: v for k, v in vars(r).items() if k not in _STD})  # fields passed via extra={...}
        if r.exc_info:
            out["exc"] = self.formatException(r.exc_info)
        return json.dumps(out, default=str)  # ASCII-escaped: the Windows console mangles UTF-8


def setup() -> None:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO), handlers=[h], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Ollama call is noise
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)  # posthog version-mismatch spam; telemetry is off
