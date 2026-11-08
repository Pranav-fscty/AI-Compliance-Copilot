"""Legal-text chunking: split on '## ' headings, then cap each section at CFG['chunk_size'] tokens.

Token counts use the embedding model's own tokenizer, so no chunk is silently truncated at embed time.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.config import CFG, chunks_file, path

logger: logging.Logger = logging.getLogger("compliance-copilot.chunk")

HEADING_PATTERN: str = r'\n(?=## )'
ENCODING: str = "utf-8"


@functools.lru_cache(maxsize=1)
def get_splitter() -> RecursiveCharacterTextSplitter:
    if CFG["chunk_strategy"] != "heading_then_tokens":
        raise ValueError(f"Unsupported chunk_strategy: {CFG['chunk_strategy']}")
    tok = AutoTokenizer.from_pretrained(CFG["chunk_tokenizer"] or CFG["embedding_model"])
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tok, chunk_size=CFG["chunk_size"], chunk_overlap=CFG["chunk_overlap"])


def chunk_by_section(text: str, source: str, law_type: str) -> list[dict]:
    """Chunk one document: '## ' sections, each split to fit the embedder window.

    Returns chunk dicts with keys: id, text, section, source, law_type, chunk_index.
    """
    chunks: list[dict] = []
    for section in re.split(HEADING_PATTERN, text):
        lines = section.strip().split("\n")
        heading = lines[0].replace("## ", "").strip()
        body = "\n".join(lines[1:]).strip()
        if len(body) < CFG["min_chunk_chars"]:
            continue
        for piece in get_splitter().split_text(body):
            chunks.append({
                "id": f"{source}#{len(chunks)}",  # stable across rebuilds: chunking is deterministic
                "text": piece.strip(),
                "section": heading or "Preamble",
                "source": source,
                "law_type": law_type,
                "chunk_index": len(chunks),
            })
    return chunks


def chunk_domain(domain: str, processed_dir: Path) -> list[dict]:
    """Chunk all documents of a domain, drop exact-duplicate texts, write chunks_file(domain)."""
    input_path = processed_dir / f"{domain}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Processed file not found: {input_path}. Run src/ingest.py first.")
    docs: list[dict] = json.loads(input_path.read_text(encoding=ENCODING))

    seen: set[str] = set()
    all_chunks: list[dict] = []
    for doc in docs:
        for c in chunk_by_section(doc["text"], doc["source"], doc["law_type"]):
            if c["text"] not in seen:  # boilerplate repeated across directions
                seen.add(c["text"])
                all_chunks.append(c)

    output_path = chunks_file(domain)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding=ENCODING)
    logger.info("Domain '%s': %d doc(s) -> %d chunk(s) -> %s", domain, len(docs), len(all_chunks), output_path)
    return all_chunks


def main() -> None:
    """Chunk all domains and print a summary table (lengths in embedder tokens)."""
    tok = get_splitter()._length_function
    print(f"{'domain':12} | {'docs':>4} | {'chunks':>6} | {'med_tok':>7} | {'max_tok':>7}")
    for domain in CFG["domains"]:
        chunks = chunk_domain(domain, path("processed_dir"))
        n = sorted(tok(c["text"]) for c in chunks)
        print(f"{domain:12} | {len({c['source'] for c in chunks}):4} | {len(chunks):6} | {n[len(n) // 2]:7} | {n[-1]:7}")


if __name__ == "__main__":
    main()
