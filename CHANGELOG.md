# Changelog

Every number below comes from `python -m eval.run` on the frozen set `eval/gold_v1.jsonl`
(74 items, sha256 `5dc17436fce5f83c…`) with the same judge (qwen3.5:4b, temp 0, seed 0), logged to MLflow
experiment `xray-target-eval`. All 74 gold items are still `claude-draft`; none have been human-verified yet.

## Post-release: Docker Compose stack run end to end

2026-11-18, Docker Desktop 29.9.2, `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`:
- All four services (api, chroma, ollama, mlflow) start and pass their healthchecks. `src.chunk` + `src.embed_store`
  run inside the api container indexed 1804 chunks into the chroma service (`compliance_bge_small_c512`) in 146 s.
- `/health` 200 `{"status":"ok","index":"ok","llm":"ok"}`. `/ask` it01 returns the same answer text as the local
  llama3.1 eval run; an out-of-corpus question returns the refusal template with no citations.
- **GPU:** it first looked like Ollama ignored the GPU. The real cause: `docker compose run`/`exec` without the
  `-f docker-compose.gpu.yml` override recreated the ollama container from the base file, which has no GPU
  (`DeviceRequests: null`). With the override on every command (`COMPOSE_FILE`, see README), Ollama detects the
  RTX 4050 (CUDA, 6 GiB) and runs llama3.1 ~89% on GPU. Short answer: 4.6 s warm (8.3 s on CPU). The first request
  after a restart takes ~170 s, because the model is loaded from the Windows models folder through the WSL mount.
- MLflow only checked as healthy; no eval run was logged to it.

## Post-release: gold set human-checked

2026-11-18: the owner checked the gold_v2 reference answers by hand and changed none of them. `eval/gold_v2.jsonl`
was not edited (its `author` fields still say `claude-draft`) so its hash, and every logged run, stays comparable.

## Post-release: LLM benchmark completed

`python -m eval.bench llms` on the default retriever (bge-small-en-v1.5, 512-token chunks, top_k 8), gold_v2, judge
qwen3.5:4b (temp 0, seed 0), no domain filter. Retrieval is identical across the three runs, so only the LLM differs.
MLflow runs `bench_llm_llama3.1`, `bench_llm_qwen3-8b`, `bench_llm_qwen3.5-4b`; numbers in `eval/bench_llms.json`.

| | llama3.1 (default) | qwen3:8b | qwen3.5:4b |
|---|---|---|---|
| correct rate (≥4) | 0.500 (32/64) | 0.563 (36/64) | 0.531 (34/64) |
| correctness (1–5) | 3.34 | 3.52 | 3.52 |
| faithfulness | **0.813 (52/64)** | 0.750 (48/64) | 0.641 (41/64) |
| false refusal rate | 0.125 | 0.125 | 0.141 |
| refusal accuracy (ooc) | 1.0 | 1.0 | 1.0 |
| /ask p50 / p95 | 3.8 s / 9.7 s | 5.2 s / 18.2 s | 2.0 s / 5.8 s |
| judge parse failures | 0 | 0 | 0 |

- **Default stays llama3.1.** It is the most faithful to the retrieved text, which matters most for compliance answers.
  The qwen models score 2–4 more correct answers out of 64, which is within noise for one judge pass and an
  unverified gold set.
- qwen3.5:4b is also the judge, so its correctness score is likely biased upward. Its faithfulness is the lowest
  despite that bias.
- qwen3:8b is the slowest (p95 almost 2x llama3.1). qwen3.5:4b is the fastest.
- This run used top_k 8. The `small_llm` preset (qwen3.5:4b, top_k 4) itself still has no answer eval.
- Single run per model: judge variance not measured.

## Post-release: default switched to bge-small-en-v1.5 (512-token chunks)

Before/after on the same frozen set (`gold_v2`), same judge (qwen3.5:4b), same LLM (llama3.1), no domain filter.
"Before" = the Phase 2 answers re-judged on gold_v2 (MLflow `default_goldv2`); "after" = MLflow `bench_llm_llama3.1`.

| | before: MiniLM-L6 (256) | after: bge-small-en-v1.5 (512) |
|---|---|---|
| hit@1 / hit@5 / hit@8 | 0.547 / 0.750 / 0.813 | 0.672 / 0.813 / 0.828 |
| MRR@10 | 0.651 | 0.732 |
| multi-section full coverage@5 | 0.286 | 0.429 |
| correctness (1–5) | 3.22 | 3.34 |
| correct rate (≥4) | 0.484 | 0.500 |
| faithfulness | 0.750 | 0.813 |
| refusal accuracy (ooc) | 1.0 | 1.0 |
| false refusal rate | 0.188 | 0.125 |
| /ask p50 / p95 | 2.1 s / 8.8 s | 3.8 s / 9.7 s |
| prompt tokens p50 / max | 740 / 1402 | 1611 / 3157 |

- Each answer-quality gap is small (1–4 questions out of 64), but all move in the same direction and none regresses. The retrieval gain is larger (hit@1 +8 questions).
- Cost: about 1.7 s slower at p50, because 512-token chunks make prompts about 2x longer. Still well inside `num_ctx` 8192.
- Re-judging the same Phase 2 answers on gold_v2 moved faithfulness from 0.734 to 0.750 (one item), which is roughly the judge's noise floor from the 4 changed gold quotes.
- The old pipeline is preserved as preset `minilm`; `bge_small` preset removed (now the default).

## Post-release: gold set re-check (gold_v2)

The user asked me to check the gold items myself instead of verifying them by hand. **This is a self-check by the same model that drafted them, not a human review**; items carry `"checked": "claude-recheck-2026-11-08"` and `author` stays `claude-draft`.

Method: for every item, the reference answer was read against ~650 characters of the raw source around its gold quote; every quote was counted across the whole raw corpus.

- **All 64 in-corpus reference answers are supported by the source text.** Two looked wrong at first and were confirmed correct: g13 (28% for HS 8415 is stated in paragraph 2.1–2.2) and r22 (the 90-day NPA norm with the 180→90-day glide path to 31 March 2026 is RBI-6 Para 44/51).
- **All 10 out-of-corpus items are absent from the corpus** (already enforced by the runner).
- **4 gold quotes were ambiguous:** each also matches a passage about something else, which could award a retrieval hit for the wrong section. Fixed in `eval/gold_v2.jsonl` (v1 left frozen):
  - it02 `It extends to the whole of India` also appears in the CGST Act → quote now includes `This Act may be called the Income-tax Act, 2025.`
  - g03 and m01 `within thirty days from the date on which he becomes liable to registration` also appears in CGST s.18 (input tax credit) → quote now starts at `in which he is so liable`, only in s.25.
  - g11 the aggregate-turnover quote also matches the definition of "turnover in State" → extended to `..., exempt supplies, exports of goods or services or both`.
- **Effect:** retrieval metrics are identical on v1 and v2 for the `default` and `bge_small` presets, so these ambiguities did not inflate any reported number.
- Other quotes occurring twice are fine: copies in the two excluded mislabeled files (not indexed), or a second occurrence on the same topic (g02 s.10(2A), g08 Removal of Difficulty orders quoting s.107).
- Runner: default gold is now `gold_v2.jsonl` (`--gold eval/gold_v1.jsonl` reproduces old runs). A cached judge verdict is reused only if the judge model **and** the gold-file hash match, because the judge prompt includes the gold quotes.

## v2.0-xray-target (2026-11-08)

Summary of the refinement (Phases 0–5). Detail per phase below.

**What changed, in one line each**
- The eval is real now: frozen 74-item verbatim-quote gold set, no domain hint in the headline, a judge from a different model family, all runs in MLflow. The old 85% and the experiment table are gone.
- The pipeline is fixed: chunks fit the embedder, the LLM context fits the prompt, mislabeled files are excluded, ingest no longer corrupts headings, retrieval is deterministic.
- Everything is configurable (`config.yaml` < preset < `RAG_*`) and separately deployable (api / chroma / ollama / mlflow).
- Ground truth for RAG X-Ray: `python -m src.manifest`; the API is black-box by default.

**Not done / not verified (read before relying on this release)**
1. ~~Gold set not human-verified~~ Resolved after release: checked by the owner on 2026-11-18, no changes (see above).
2. ~~LLM benchmark not run~~ Resolved after release: all three LLMs evaluated, llama3.1 kept (see above).
3. ~~Default config not changed~~ Resolved after release: bge-small (512) evaluated end to end and made the default (see above).
4. ~~Docker Compose stack never run end to end~~ Resolved after release: runs and answers on GPU (see above).
5. `small_llm` and `nomic` presets have no answer eval.
6. Income Tax coverage is still only Chapters I–III. `data/raw/` is not in git.
7. `/compare` and `/simplify` prompts are in code and have no eval. `reranker` accepts only `none`.
8. `docs/architecture.png` shows the old single-container design and is no longer referenced by the README.

## Phase 4: separate services, presets, ground truth, black-box API

- **docker-compose with 4 services:** `api`, `chroma` (server mode, `chromadb/chroma:0.5.3` = client pin), `ollama`, `mlflow` (v2.16.0 = client pin), each with a healthcheck; `api` waits for healthy `chroma` and `ollama`. Ollama reuses the host's models folder; `docker-compose.gpu.yml` adds the NVIDIA GPU.
  - **Not run end to end.** Docker Desktop's engine would not start on this machine (its log shows a failed Docker sign-in token refresh). What was verified: both compose files pass `docker compose config`; Chroma server mode was tested outside Docker with `chroma run` (same retrieval hit@k as embedded mode; MRR differs in the 4th decimal because the HNSW graph was built in another process).
- **Presets** selected by `RAG_PRESET`: `default` (Phase 2 pipeline, fully evaluated), `bge_small` (best retriever, retrieval-only eval), `nomic` (Ollama embedder, top_k 5), `small_llm` (qwen3.5:4b, top_k 4, not evaluated). Precedence: `config.yaml` < preset < `RAG_*` env. Unknown preset names, unknown preset keys, or misspelled `RAG_*` variables raise an error.
- **Ground truth:** `python -m src.manifest` writes `ground_truth/pipeline_manifest.json` for the active preset from config plus live facts (chunk counts and token lengths, collection size, metric and `search_ef`, embedding dimension, Ollama model digests, prompt and its sha256, git commit + dirty flag). The API never reads it; `ground_truth/` is gitignored and excluded from the image.
- **Black-box API:** `/docs`, `/redoc` and `/openapi.json` are off. `POST /debug/retrieve` (and the docs) exist only with `debug_endpoints: true`; default false.
- **Structured logs:** one JSON object per line with `request_id`. The id is taken from a client `X-Request-ID` if it matches `[A-Za-z0-9._-]{1,64}`, otherwise generated, and returned in the `X-Request-ID` response header. One `request` line per call with method, path, status, duration_ms. uvicorn's own access log is off.
- **Failure behaviour (checked):** Chroma down: `/health` 503 `{"index":"error"}`, `/ask` 500. No fault-injection hooks exist in the app; faults are meant to be injected from outside (stop/slow/corrupt a container).
- Chunk files moved to `data/processed/chunks/<collection>/` so presets with different chunking don't overwrite each other; they are derived and no longer committed.
- Eval runner honours `MLFLOW_TRACKING_URI` (to log into the compose `mlflow` service); a Chroma server has no local SQLite file, so `chroma_sqlite_sha` is only logged in embedded mode.

## Phase 3: embedder and LLM benchmark

Command: `python -m eval.bench embedders` (each candidate chunked and indexed into its own Chroma collection,
then `eval.run --retrieval-only`). Results are in `eval/bench_embedders.json` and in MLflow runs `bench_*_retrieval`.
Same frozen 64 in-corpus questions, no metadata filter.

| Embedder (chunk tokens) | hit@1 | hit@3 | hit@5 | hit@8 | MRR@10 | multi full cov@5 | retrieve p50 |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (256), Phase 2 default | 0.547 | 0.734 | 0.750 | 0.813 | 0.651 | 0.286 | 18 ms |
| bge-small-en-v1.5 (256) | 0.641 | 0.734 | 0.766 | 0.813 | 0.698 | 0.429 | 30 ms |
| **bge-small-en-v1.5 (512)** | **0.672** | **0.781** | 0.813 | 0.828 | **0.732** | **0.429** | 29 ms |
| bge-base-en-v1.5 (256) | 0.594 | 0.750 | 0.813 | 0.844 | 0.689 | 0.286 | 70 ms |
| bge-base-en-v1.5 (512) | 0.641 | 0.766 | **0.828** | **0.891** | 0.720 | 0.286 | 61 ms |
| nomic-embed-text via Ollama (500) | 0.625 | 0.797 | 0.797 | 0.859 | 0.721 | 0.429 | 20 ms |
| mxbai-embed-large via Ollama (512) | 0.547 | 0.766 | 0.813 | 0.875 | 0.666 | 0.286 | 26 ms |

Chunk size is always at or below the embedder's window, measured with its own tokenizer (max 504 for bge at 512).
Retrieve latency includes the query embedding; HF models run on CPU (torch CPU build), Ollama embedders on the GPU.

**Reading it honestly:**
- One question = 1.6 points, so most gaps between the top four are 1-3 questions and could be noise. The clear result: every candidate beats MiniLM on hit@1/MRR, and 512-token chunks beat 256 for the same bge model.
- bge-small (512) is the best on ranking (hit@1, MRR) and 33M params; bge-base (512) is best on hit@5/8 but 3x larger and 2x slower on CPU.

**LLM benchmark: not completed in Phase 3** (completed after release, see the top of this file). `python -m eval.bench llms` (llama3.1 vs qwen3:8b vs qwen3.5:4b, full eval on the bge-small retriever) was killed for low system memory after 9 of 74 generations; there are no LLM comparison numbers. Resume with the same command (checkpoints are kept). Known issue for it: qwen3.5:4b is also the judge, so its own scores would be biased upward.

**Decision:** the default stays the Phase 2 config, because the bge-small retriever has no end-to-end answer eval yet and the ground rule is no embedder/chunking change without one. bge-small (512) ships as preset `bge_small` with a note that it is the recommended next default.

### Changed in Phase 3
- `embedding_backend` (`hf` | `ollama`), `embed_query_prefix` / `embed_doc_prefix` (bge / nomic need them), `chunk_tokenizer` config keys. The embedder is a small in-house wrapper over sentence-transformers / Ollama `/api/embed` instead of the deprecated `langchain_community` HuggingFaceEmbeddings; vectors are identical for MiniLM (retrieval metrics unchanged to the last digit).
- The LLM is called through the `ollama` client instead of `langchain-ollama`'s ChatOllama, so `think=false` can be passed (qwen3 models otherwise emit a reasoning trace); `langchain-ollama` dropped from requirements.
- Fixed: `RAG_EMBED_QUERY_PREFIX="search_query: "` was parsed by YAML into a dict; string-valued keys now keep the raw env string.
- Fixed: `/ask` returned all retrieved chunks as citations on a refusal; refusals now return no citations.

## Phase 2: fixes and a single config

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

### Still open after Phase 2 (see v2.0 summary for the current list)
- `/compare` and `/simplify` prompts still live in code, not config.
- Citation headings are only partly cleaned (cosmetic).
- Income Tax coverage is still only Chapters I–III of the Act.
- Gold set human verification: 0 of 74 items done.
- `reranker` is a config key, but the only allowed value is `none`.
- Still on the deprecated `langchain_community` Chroma wrapper (the HF embeddings wrapper was replaced in Phase 3).
