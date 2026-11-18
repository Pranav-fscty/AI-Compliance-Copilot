"""Write ground_truth/pipeline_manifest.json: the true pipeline for the active preset.

    python -m src.manifest                  # active preset (RAG_PRESET, default "default")
    RAG_PRESET=fast python -m src.manifest

This is the answer key RAG X-Ray's guesses are scored against. The API never reads or serves it,
and ground_truth/ is excluded from the Docker image.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time

import ollama

from src.config import CFG, PRESET, ROOT, chunks_file

OUT = ROOT / "ground_truth" / "pipeline_manifest.json"
REFUSAL = "This information is not available in the indexed documents."


def _try(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a down service becomes null + reason, not a crash
        return {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}


def _ollama_digest(name: str):
    for m in ollama.Client(host=CFG["ollama_base_url"]).list()["models"]:
        if m["model"] in (name, f"{name}:latest"):
            return m["digest"]
    return None


def _chunk_stats() -> dict:
    from src.chunk import get_splitter
    tok = get_splitter()._length_function
    out = {}
    for d in CFG["domains"]:
        chunks = json.loads(chunks_file(d).read_text(encoding="utf-8"))
        n = sorted(tok(c["text"]) for c in chunks)
        out[d] = {"chunks": len(chunks), "sources": sorted({c["source"] for c in chunks}),
                  "tokens_median": statistics.median(n), "tokens_max": n[-1]}
    return out


def _index() -> dict:
    from src.embed_store import get_client, get_embeddings
    col = get_client().get_collection(CFG["collection"])
    return {"collection": CFG["collection"], "count": col.count(), "metadata": col.metadata,
            "embedding_dim": len(get_embeddings().embed_query("dimension probe"))}


def build() -> dict:
    git = _try(lambda: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git,
        "git_dirty": _try(lambda: bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())),
        "preset": PRESET,
        "pipeline": {
            "ingest": {"source_format": "markdown", "domains": CFG["domains"], "excluded_sources": CFG["exclude_sources"]},
            "chunking": {"strategy": CFG["chunk_strategy"], "size_tokens": CFG["chunk_size"], "overlap_tokens": CFG["chunk_overlap"],
                         "tokenizer": CFG["chunk_tokenizer"] or CFG["embedding_model"], "min_section_chars": CFG["min_chunk_chars"]},
            "embedding": {"backend": CFG["embedding_backend"], "model": CFG["embedding_model"], "normalized": True,
                          "query_prefix": CFG["embed_query_prefix"], "doc_prefix": CFG["embed_doc_prefix"],
                          "ollama_digest": _try(lambda: _ollama_digest(CFG["embedding_model"])) if CFG["embedding_backend"] == "ollama" else None},
            "vector_store": {"engine": "chroma", "version": "0.5.3", "distance": CFG["distance"], "index": "hnsw",
                             "hnsw_search_ef": CFG["hnsw_search_ef"], "mode": "server" if CFG["chroma_host"] else "embedded"},
            "retrieval": {"top_k": CFG["top_k"], "metadata_filter": "law_type, only when the client passes it", "hybrid": False,
                          "query_rewriting": False, "reranker": CFG["reranker"],
                          "rerank_fetch_k": CFG["rerank_fetch_k"] if CFG["reranker"] != "none" else None},
            "generation": {"llm": CFG["llm_model"], "llm_digest": _try(lambda: _ollama_digest(CFG["llm_model"])),
                           "temperature": CFG["temperature"], "num_ctx": CFG["num_ctx"], "think": CFG["llm_think"],
                           "prompt_template": CFG["prompt_template"],
                           "prompt_sha256": hashlib.sha256(CFG["prompt_template"].encode()).hexdigest(),
                           "context_order": "retrieval rank, chunks joined by blank lines, no headings",
                           "refusal_string": REFUSAL},
            "citations": "(source, section) of every chunk put in the prompt, deduplicated; empty list when the answer is the refusal",
            "api": {"endpoints": ["POST /ask", "POST /compare", "POST /simplify", "GET /health"],
                    "debug_endpoints_enabled": CFG["debug_endpoints"]},
        },
        "chunks": _try(_chunk_stats),
        "index": _try(_index),
    }


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
