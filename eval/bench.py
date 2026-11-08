"""Phase 3 benchmark: each candidate = RAG_* overrides, built into its own Chroma collection.

    python -m eval.bench embedders          # chunk + embed + retrieval-only eval per embedder candidate
    python -m eval.bench llms               # full eval (gen + judge) per LLM on the chosen retriever
    python -m eval.bench embedders bge-small_c256   # just one candidate

Each step is a subprocess because src.config reads the environment once at import.
Note: chunking rewrites data/processed/*_chunks.json; re-run `python -m src.chunk` for the default config afterwards.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BGE_Q = "Represent this sentence for searching relevant passages: "

EMBEDDERS = {
    "minilm_c256": {},  # Phase 2 default
    "bge-small_c256": {"EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBED_QUERY_PREFIX": BGE_Q, "CHUNK_SIZE": "256", "CHUNK_OVERLAP": "32"},
    "bge-small_c512": {"EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBED_QUERY_PREFIX": BGE_Q, "CHUNK_SIZE": "512", "CHUNK_OVERLAP": "64"},
    "bge-base_c256": {"EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5", "EMBED_QUERY_PREFIX": BGE_Q, "CHUNK_SIZE": "256", "CHUNK_OVERLAP": "32"},
    "bge-base_c512": {"EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5", "EMBED_QUERY_PREFIX": BGE_Q, "CHUNK_SIZE": "512", "CHUNK_OVERLAP": "64"},
    "nomic_c512": {"EMBEDDING_BACKEND": "ollama", "EMBEDDING_MODEL": "nomic-embed-text", "CHUNK_TOKENIZER": "nomic-ai/nomic-embed-text-v1.5",
                   "EMBED_QUERY_PREFIX": "search_query: ", "EMBED_DOC_PREFIX": "search_document: ", "CHUNK_SIZE": "500", "CHUNK_OVERLAP": "64"},
    "mxbai_c512": {"EMBEDDING_BACKEND": "ollama", "EMBEDDING_MODEL": "mxbai-embed-large", "CHUNK_TOKENIZER": "mixedbread-ai/mxbai-embed-large-v1",
                   "EMBED_QUERY_PREFIX": BGE_Q, "CHUNK_SIZE": "512", "CHUNK_OVERLAP": "64"},
}
LLMS = ["llama3.1", "qwen3:8b", "qwen3.5:4b"]


def run(args: list[str], overrides: dict) -> str:
    env = {**os.environ, "ANONYMIZED_TELEMETRY": "False", "PYTHONIOENCODING": "utf-8",
           **{f"RAG_{k}": v for k, v in overrides.items()}}
    p = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"{args} failed:\n{p.stderr[-3000:]}")
    return p.stdout


def metrics(stdout: str) -> dict:
    return json.loads(stdout[stdout.index('{\n  "params"'):])["metrics"]


def main() -> None:
    what, only = sys.argv[1], sys.argv[2:]
    rows = {}
    if what == "embedders":
        for name, ov in EMBEDDERS.items():
            if only and name not in only:
                continue
            ov = {**ov, "COLLECTION": f"bench_{name}".replace("-", "_")}
            print(f"== {name}: chunk + embed", flush=True)
            run(["src.chunk"], ov)
            run(["src.embed_store"], ov)
            rows[name] = metrics(run(["eval.run", "--retrieval-only", "--run-name", f"bench_{name}"], ov))
            print(json.dumps({name: rows[name]}), flush=True)
    elif what == "llms":
        base = json.loads(os.environ.get("BENCH_RETRIEVER", "{}"))  # overrides of the chosen retriever
        for llm in LLMS:
            if only and llm not in only:
                continue
            print(f"== {llm}: full eval", flush=True)
            rows[llm] = metrics(run(["eval.run", "--run-name", f"bench_llm_{llm}"], {**base, "LLM_MODEL": llm}))
            print(json.dumps({llm: rows[llm]}), flush=True)
    out = ROOT / "eval" / f"bench_{what}.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
