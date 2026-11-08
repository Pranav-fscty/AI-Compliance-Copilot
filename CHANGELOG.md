# Changelog

Every number below comes from `python -m eval.run` on the frozen set `eval/gold_v1.jsonl`
(74 items, sha256 `5dc17436fce5f83c…`) with the same judge (qwen3.5:4b, temp 0, seed 0), logged to MLflow
experiment `xray-target-eval`. All 74 gold items are still `claude-draft`; none have been human-verified yet.

## Unreleased: Phase 2 (fixes and a single config)

### Before / after

`baseline_as_shipped` = the repo exactly as it was at `e257b36`. `phase2_fixed` = this branch.
Retrieval headline is **no metadata filter**; "oracle" passes the gold `law_type` as a filter.

| Metric | baseline_as_shipped | phase2_fixed | Δ |
|---|---|---|---|
| hit@1 (no filter) | 0.594 | 0.547 | **−0.047** |
| hit@3 | 0.703 | 0.734 | +0.031 |
| hit@5 | 0.734 | 0.750 | +0.016 |
| hit@8 | 0.766 | 0.813 | +0.047 |
| MRR@10 | 0.653 | 0.651 | −0.002 |
| hit@8 (oracle filter) | 0.766 | 0.844 | +0.078 |
| multi-section full coverage@5 | 0.286 | 0.286 | 0 |
| correctness mean (1–5) | 2.75 | 3.22 | +0.47 |
| correct rate (≥4) | 0.375 | 0.484 | +0.109 |
| faithfulness rate | 0.609 | 0.734 | +0.125 |
| refusal accuracy (10 ooc) | 1.00 | 1.00 | 0 |
| false refusal rate (64 in-corpus) | 0.203 | 0.188 | −0.016 |
| judge parse failures | 0 | 0 | |
| /ask latency p50 / p95 (s) | 3.34 / 11.76 | 2.10 / 8.83 | faster |
| prompt tokens p50 / max | 2053 / 44143 | 740 / 1402 | |
| prompts over num_ctx | 20.3% (of 4096) | 0% (of 8192) | |

Caveats:
- n=64 in-corpus questions, so one question is 1.6 points. Only the faithfulness, correctness and hit@8 changes are clearly bigger than noise; the hit@1 drop is about 3 questions and real. Smaller chunks rank more fragments of the same section above the gold one.
- The baseline retrieval numbers come from a single restart of a nondeterministic index (6-restart range for hit@5 was 0.719–0.766). Phase 2 retrieval is deterministic.
- Several changes land together (chunking, num_ctx, file exclusion, ingest fixes, dedupe), so this table cannot attribute the gain to any one of them.
- At `num_ctx` 8192, llama3.1 8B no longer fits in 6 GB VRAM (`ollama ps`: 32% CPU / 68% GPU). Latency still dropped because prompts are about 3× shorter.
- Judge = qwen3.5:4b, a small model. Absolute correctness numbers are only good for comparing runs, not as truth.

### Fixed

**P0: broken or incorrect**
- **The LLM context window was never set.** Ollama ran with `num_ctx` 4096. 20% of eval prompts were longer than that (max 44k tokens) and were silently truncated, often cutting the "use only the context" instruction. Now `num_ctx: 8192` comes from config, and the runner logs prompt token counts and the share over budget.
- **Chunks were bigger than the embedder could read.** Splitting only on `##` produced chunks up to 22,480 tokens, but MiniLM reads 256. 28% of chunks were truncated, and 46.8% of corpus tokens were never embedded. Now each heading section is split with the embedder's own tokenizer (256 tokens, 32 overlap), so the max is 255 tokens.
- **Two "GST circulars" were Income-tax Act text.** `Circular_No__236302024_GST.md` and `…237312024_GST.md` contain Income-tax Act Chapters I–III, a duplicate of the income_tax file. They were indexed as one 68k and one 98k character chunk tagged `gst`. They are now excluded via `exclude_sources` in config (not deleted).
- **Processed data could not be rebuilt from raw.** `data/processed/income_tax.json` had been hand-edited. Running ingest + chunk gave 17 chunks with different headings, not the shipped 14. The hand edit is gone; everything is regenerated from `data/raw` (the old file is still in git history). Two bugs in `ingest.clean_text` caused the heading damage:
  - the line-join regex glued the body line after a heading onto the heading;
  - the repeated-short-line filter (meant for page headers) also deleted numbered headings like `## 1.` that legitimately repeat across annexes, which merged whole sections.
- **The headline 85% was not a real result.** It was hit@8 with the gold `law_type` passed as a filter, on 20 self-written definitional questions, with gold labels edited after seeing retrieval output (I reproduced exactly 17/20). Replaced by the frozen 74-item set: span-based gold quoted from `data/raw`, no filter in the headline, 10 out-of-corpus questions, and a judge from a different model family that sees the reference answer.
- **The README experiment grid was fabricated.** All 18 MLflow grid runs had status FAILED, ran under 1 s and logged 0 metrics. The one run with metrics was written by `log_baseline.py` with typed-in constants. The table is removed. That run is tagged `source=manual_unverified` in MLflow and kept.
  - Root cause of the failures: on Windows, importing mlflow (pyarrow) before torch makes torch's `c10.dll` fail to load (WinError 1114). The project notes blamed "subprocess context", which was wrong. `eval/run.py` imports torch first.
- **Docker could not reach Ollama and could not build from a clone.** The container used `localhost:11434`, and the image `COPY`'d a gitignored `chroma_db/`. Now Docker uses `RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434`, `chroma_db` is a volume, and the HF cache is a named volume. `env_file` is removed, so the unused OpenAI key in `.env` no longer goes into the container. *The Docker build has not been tested yet (Phase 4 rebuilds compose anyway).*

**P1: inconsistent or fragile**
- **Unneeded OpenAI key and dependencies.** `config.py` refused to start without `OPENAI_API_KEY`, which nothing used. Removed, along with unused deps (`langchain-openai`, `pypdf`, `unstructured`, `pdfplumber`, `markitdown`). Every direct dependency is pinned in `requirements.txt` (torch, transformers and tokenizers were unpinned), and `requirements.lock` holds the full `pip freeze`. The Dockerfile installs from the lock.
- **Settings were spread across 5 files.** Now `config.yaml` holds chunking (strategy, size, overlap), embedder, distance, `search_ef`, top_k, reranker, prompt template, LLM, `num_ctx`, temperature and Ollama URL. Each key can be overridden with `RAG_<KEY>`; a misspelled `RAG_*` variable raises an error instead of being ignored. Eval runs log the whole config to MLflow.
- **Retrieval was not deterministic between restarts.** Chroma rebuilds HNSW from its WAL on every open. With the default `search_ef=10`, hit@k varied between restarts (hit@5 ranged 0.719–0.766 over 6 restarts). Now `hnsw:search_ef: 100` is set; 3 consecutive fresh-process runs gave identical retrieval metrics.
- **Distance metric and chunk IDs were implicit.** The metric is now set explicitly (`hnsw:space: cosine`), and embeddings are explicitly normalized. Chunk IDs are stable (`<source>#<n>`) instead of random UUIDs on every rebuild.
- **Experiments could wipe the live index.** `src/experiments.py` called `rmtree` on the live `chroma_db` and monkeypatched module globals. It is deleted. The build now drops and recreates only `CFG["collection"]`, so several collections (embedders, chunk sizes) can live side by side.
- **`/health` hung while an answer was generating.** Handlers were `async def` but made blocking calls with one worker, so `/health` stalled. They are now plain `def` (FastAPI thread pool). `/health` checked nothing; it now checks that the collection is non-empty and that Ollama answers, and returns 503 `degraded` otherwise. It does not expose model names.
- **No MLflow tracking URI was set.** The URI is now explicit, absolute and file-based, and the eval logs itself. `src/eval.py`, `log_baseline.py` and `eval/questions.json` are deleted (superseded by `eval/run.py` + `eval/gold_v1.jsonl`).
- **Duplicate chunks.** Exact duplicate texts (boilerplate repeated across RBI directions) are dropped at chunk time.

**P2 / housekeeping**
- `DOMAINS` was defined 3 times; it is now `CFG["domains"]`. `ingest`/`chunk` no longer force DEBUG logging; use `LOG_LEVEL`.
- `rag_chain` now uses the embedder from `embed_store` instead of building a second one, and calls the LLM directly instead of through `RetrievalQA`. The prompt is identical to before.
- README: removed "production-ready" and the fake results table, corrected the offline and grounding claims, removed `ollama serve` (it conflicts with the Windows service), and documented `config.yaml`. `.env.example` now exists.

### Corrections to my own Phase 0 audit
- I said citations were "not the ones the model used". In fact citations = exactly the chunks put into the prompt. They are not filtered to what the answer relied on, but they aren't wrong either.
- Some of the junk in section headings comes from the PDF→Markdown conversion in `data/raw` itself, not only from the ingest bugs.
- Chroma's "Failed to send telemetry event" log line is a posthog version mismatch. With `anonymized_telemetry=False` the client is disabled and nothing is sent.

### Still open (not fixed)
- `/compare` and `/simplify` prompts still live in code, not config.
- Citation headings are only partly cleaned (cosmetic).
- `/ask` returns all retrieved chunks as citations even when the answer is the refusal string (found in the Phase 2 smoke test).
- Income Tax coverage is still only Chapters I–III of the Act.
- Gold set human verification: 0 of 74 items done.
- `reranker` is a config key, but the only allowed value is `none`.
- Still on the deprecated `langchain_community` Chroma/HF wrappers.
