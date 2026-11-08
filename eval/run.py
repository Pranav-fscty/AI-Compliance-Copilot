"""Frozen-set evaluation runner.

    python -m eval.run                    # retrieval (none + oracle filter) + generation + judge
    python -m eval.run --retrieval-only   # fast, deterministic retrieval metrics only

Gold = verbatim quotes from data/raw. A retrieved chunk is a hit if it contains the
(normalised) quote, or a contiguous overlap >= max(40, 60% of the quote). This is
chunker-independent, so the same gold works after re-chunking / re-embedding.
Generation goes through the app's own rag_chain.ask(); nothing in src/ is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch  # noqa: F401  must load before mlflow/pyarrow on Windows, else WinError 1114 in c10.dll
import mlflow
import ollama
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "gold_v1.jsonl"
RAW = ROOT / "data" / "raw"
EXPERIMENT = "xray-target-eval"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen3.5:4b")
REFUSAL = "not available in the indexed documents"
RANK_K = 10
JUDGE_CHUNK_CHARS = 2500  # shortcut: judge sees each retrieved chunk cut to 2.5k chars (fits num_ctx 8192); faithfulness on longer chunks may be under-credited

log = logging.getLogger("eval.run")


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[―‖“”‘’\"'`*#]", " ", s)
    s = re.sub(r"[–—]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_hit(quote_n: str, doc_n: str) -> bool:
    if quote_n in doc_n:
        return True
    m = SequenceMatcher(None, quote_n, doc_n, autojunk=False).find_longest_match(0, len(quote_n), 0, len(doc_n))
    return m.size >= max(40, 0.6 * len(quote_n))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_gold() -> list[dict]:
    items = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    raw = {p.name: norm(p.read_text(encoding="utf-8")) for p in RAW.glob("*/*.md")}
    corpus = " ".join(raw.values())
    errors = []
    for it in items:
        for g in it["gold"]:
            g["_n"] = norm(g["quote"])
            if g["_n"] not in raw.get(g["source"], ""):
                errors.append(f"{it['id']}: quote not in {g['source']}: {g['quote'][:60]}")
        for t in it.get("absent_terms", []):
            if norm(t) in corpus:
                errors.append(f"{it['id']}: out-of-corpus term present in corpus: {t}")
    assert len({i["id"] for i in items}) == len(items), "duplicate ids"
    if errors:
        sys.exit("Gold set invalid:\n  " + "\n  ".join(errors))
    return items


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))] if xs else float("nan")


def retrieval_metrics(items, rc, mode: str) -> tuple[dict, dict]:
    per, lat = {}, []
    for it in items:
        if it["category"] == "ooc":
            continue
        law = it["law_type"] if mode == "oracle" else None
        t0 = time.perf_counter()
        docs = rc.retrieve(it["question"], law, k=RANK_K)  # the app's own retrieval path
        lat.append((time.perf_counter() - t0) * 1000)
        dn = [norm(d.page_content) for d in docs]
        ranks = [next((r for r, d in enumerate(dn, 1) if quote_hit(g["_n"], d)), None) for g in it["gold"]]
        first = min((r for r in ranks if r), default=None)
        per[it["id"]] = {"first_hit_rank": first, "gold_ranks": ranks}
    n = len(per)
    m = {f"hit_at_{k}": sum(1 for v in per.values() if v["first_hit_rank"] and v["first_hit_rank"] <= k) / n for k in (1, 3, 5, 8)}
    m["mrr_at_10"] = sum(1 / v["first_hit_rank"] for v in per.values() if v["first_hit_rank"]) / n
    multi = [v for i, v in per.items() if i.startswith("m")]
    m["multi_full_coverage_at_5"] = sum(all(r and r <= 5 for r in v["gold_ranks"]) for v in multi) / len(multi)
    m["retrieve_ms_p50"], m["retrieve_ms_p95"] = pct(lat, .5), pct(lat, .95)
    return {f"{mode}_{k}": v for k, v in m.items()}, per


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "refused": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["reason", "refused", "unsupported_claims", "correctness"],
}

JUDGE_PROMPT = """You grade answers from a legal RAG assistant. Be strict and literal.

QUESTION: {question}
REFERENCE ANSWER: {reference}
GOLD SOURCE TEXT: {gold}

RETRIEVED CONTEXT GIVEN TO THE ASSISTANT:
{context}

ASSISTANT ANSWER:
{answer}

Return JSON with, in this order:
- reason: one or two sentences of analysis first.
- refused: true ONLY if the answer declines or says the information is not available. An answer that states facts (even wrong ones) is NOT a refusal.
- unsupported_claims: every factual claim in the answer that is NOT stated in the RETRIEVED CONTEXT (empty list if all claims are supported or the answer is a refusal).
- correctness: 1-5 agreement with the REFERENCE ANSWER (5 = all key facts right, 3 = partly right, 1 = wrong, missing or refused)."""


def judge(client, it, answer, ctx_docs) -> dict:
    ctx = "\n\n".join(f"[{i}] {d[:JUDGE_CHUNK_CHARS]}" for i, d in enumerate(ctx_docs, 1))
    prompt = JUDGE_PROMPT.format(question=it["question"], reference=it["reference_answer"],
                                 gold=" | ".join(g["quote"] for g in it["gold"]) or "(none: not in corpus)",
                                 context=ctx, answer=answer)
    try:
        r = client.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], think=False,
                        format=JUDGE_SCHEMA, options={"temperature": 0, "seed": 0, "num_ctx": 8192})
        out = json.loads(r["message"]["content"])
        out["correctness"] = max(1, min(5, int(out["correctness"])))
        out["faithful"] = not out["unsupported_claims"]
        return out
    except Exception as exc:  # parse/transport failures are counted, never scored
        return {"parse_error": f"{type(exc).__name__}: {exc}"[:200]}


def model_digest(client, name: str) -> str:
    for m in client.list()["models"]:
        if m["model"] == name or m["model"] == f"{name}:latest":
            return m["digest"][:12]
    return "unknown"


def loaded_ctx(client, name: str):
    for m in client.ps()["models"]:
        if m["model"].startswith(name):
            return m.get("context_length")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--run-name", default="baseline_as_shipped")
    ap.add_argument("--fresh", action="store_true", help="discard the checkpoint for this run name")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 console crashed on '₹' in judge output

    items = load_gold()
    sys.path.insert(0, str(ROOT))
    from src.config import CFG, PRESET, chunks_file, path
    hashes = {"gold_sha": sha(GOLD), "preset": PRESET}
    if not CFG["chroma_host"]:  # hash before opening the DB; a Chroma server has no local file to hash
        hashes["chroma_sqlite_sha"] = sha(path("chroma_dir") / "chroma.sqlite3")
    for d in CFG["domains"]:
        hashes[f"chunks_{d}_sha"] = sha(chunks_file(d))
    import src.rag_chain as rc

    rc.get_db()
    client = ollama.Client(host=CFG["ollama_base_url"])
    params = {
        **{k: v for k, v in CFG.items() if k != "prompt_template"},
        "prompt_sha": hashlib.sha256(CFG["prompt_template"].encode()).hexdigest()[:16],
        "llm_digest": model_digest(client, CFG["llm_model"]),
        "judge_model": JUDGE_MODEL, "judge_digest": model_digest(client, JUDGE_MODEL),
        "n_items": len(items), "n_ooc": sum(i["category"] == "ooc" for i in items),
        "gold_authors": json.dumps(sorted({i["author"] for i in items})), **hashes,
    }

    metrics, per_item = {}, {i["id"]: {"id": i["id"], "category": i["category"]} for i in items}
    for mode in ("none", "oracle"):
        m, per = retrieval_metrics(items, rc, mode)
        metrics.update(m)
        for k, v in per.items():
            per_item[k][f"retrieval_{mode}"] = v

    if not args.retrieval_only:
        # Checkpoint: one JSON line per finished gen/judge step, so a crash or OOM kill loses at most one item.
        ckpt = ROOT / "eval" / "checkpoints" / f"{args.run_name}.jsonl"
        if args.fresh and ckpt.exists():
            ckpt.unlink()
        ckpt.parent.mkdir(exist_ok=True)
        done = {}
        if ckpt.exists():
            for line in ckpt.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["stage"] == "gen" or rec.get("judge_model") == JUDGE_MODEL:
                    done[(rec["stage"], rec["id"])] = rec

        def save(rec):
            with ckpt.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        enc = tiktoken.get_encoding("cl100k_base")
        rc.ask("warm-up question", None)  # load model; excluded from latency
        params["num_ctx_observed"] = loaded_ctx(client, CFG["llm_model"])
        for n, it in enumerate(items, 1):
            rec = done.get(("gen", it["id"]))
            if rec is None:
                retrieved = rc.retrieve(it["question"])
                docs = [d.page_content for d in retrieved]
                ptok = len(enc.encode(rc.build_prompt(it["question"], retrieved)))
                t0 = time.perf_counter()
                res = rc.ask(it["question"], None)  # headline: no law_type hint, like a black-box client
                rec = {"stage": "gen", "id": it["id"], "answer": res["answer"], "ask_s": round(time.perf_counter() - t0, 2),
                       "prompt_tokens_cl100k": ptok, "ctx": docs}
                save(rec)
            per_item[it["id"]].update(answer=rec["answer"], ask_s=rec["ask_s"], prompt_tokens_cl100k=rec["prompt_tokens_cl100k"], _ctx=rec["ctx"])
            print(f"[gen {n}/{len(items)}] {it['id']} {rec['ask_s']:.1f}s", flush=True)
        for n, it in enumerate(items, 1):  # second pass so the two models don't swap per question
            p = per_item[it["id"]]
            rec = done.get(("judge", it["id"]))
            if rec is None:
                rec = {"stage": "judge", "id": it["id"], "judge_model": JUDGE_MODEL, "judge": judge(client, it, p["answer"], p["_ctx"])}
                save(rec)
            p.pop("_ctx")
            p["judge"] = rec["judge"]
            p["refused_string"] = REFUSAL in p["answer"].lower()
            print(f"[judge {n}/{len(items)}] {it['id']} {p['judge']}", flush=True)
        totals = [per_item[i["id"]]["ask_s"] for i in items]
        ptoks = [per_item[i["id"]]["prompt_tokens_cl100k"] for i in items]

        def j(p):
            return p["judge"] if "parse_error" not in p["judge"] else None
        inc = [per_item[i["id"]] for i in items if i["category"] != "ooc"]
        ooc = [per_item[i["id"]] for i in items if i["category"] == "ooc"]
        refused = lambda p: p["refused_string"] or bool(j(p) and j(p)["refused"])
        jud = [j(p) for p in inc if j(p)]
        metrics.update({
            "correctness_mean": statistics.mean(x["correctness"] for x in jud),
            "correct_rate_ge4": sum(x["correctness"] >= 4 for x in jud) / len(jud),
            "faithfulness_rate": sum(x["faithful"] for x in jud) / len(jud),
            "refusal_accuracy_ooc": sum(refused(p) for p in ooc) / len(ooc),
            "false_refusal_rate": sum(refused(p) for p in inc) / len(inc),
            "judge_parse_failures": sum(1 for p in per_item.values() if "parse_error" in p["judge"]),
            "ask_s_p50": pct(totals, .5), "ask_s_p95": pct(totals, .95),
            "prompt_tokens_p50": pct(ptoks, .5), "prompt_tokens_max": max(ptoks),
        })
        if params["num_ctx_observed"]:
            metrics["pct_prompts_over_num_ctx"] = sum(t > params["num_ctx_observed"] for t in ptoks) / len(ptoks)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or (ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name + ("_retrieval" if args.retrieval_only else "")):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "per_item.jsonl"
            out.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in per_item.values()), encoding="utf-8")
            (Path(td) / "summary.json").write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))
            mlflow.log_artifacts(td)
    print(json.dumps({"params": params, "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
