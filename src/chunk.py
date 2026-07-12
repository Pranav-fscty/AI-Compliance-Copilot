"""Legal-text chunking pipeline for the compliance-copilot application."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger("compliance-copilot.chunk")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHUNK_CHARS: int = 50
HEADING_PATTERN: str = r'\n(?=## )'
ENCODING: str = "utf-8"
DOMAINS: tuple[str, ...] = ("income_tax", "gst", "rbi")

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PROCESSED_BASE_DIR: str = str(_PROJECT_ROOT / "data" / "processed")


# ---------------------------------------------------------------------------
# Core chunking functions
# ---------------------------------------------------------------------------

def chunk_by_section(text: str, source: str, law_type: str) -> list[dict]:
    """Chunk a document by ## Markdown headings.

    Args:
        text: Full document text with ## headings as section markers.
        source: Filename of the source document.
        law_type: Domain identifier (income_tax, gst, rbi).

    Returns:
        List of chunk dicts with keys: text, section, source, law_type.
    """
    sections = re.split(HEADING_PATTERN, text)
    chunks: list[dict] = []

    for section in sections:
        lines = section.strip().split("\n")
        heading = lines[0].replace("## ", "").strip()
        body = "\n".join(lines[1:]).strip()

        if len(body) < MIN_CHUNK_CHARS:
            logger.debug("Skipping short chunk under heading '%s' (%d chars)", heading, len(body))
            continue

        chunks.append({
            "text": body,
            "section": heading if heading else "Preamble",
            "source": source,
            "law_type": law_type,
        })

    return chunks


def chunk_domain(domain: str, processed_dir: str) -> list[dict]:
    """Load processed JSON for a domain and chunk all documents.

    Args:
        domain: One of 'income_tax', 'gst', 'rbi'.
        processed_dir: Path to data/processed/ directory.

    Returns:
        Combined list of all chunk dicts for the domain.

    Raises:
        FileNotFoundError: If the processed JSON for the domain is missing.
    """
    input_path: Path = Path(processed_dir) / f"{domain}.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {input_path}. Run src/ingest.py first."
        )

    with input_path.open(mode="r", encoding=ENCODING) as fh:
        docs: list[dict] = json.load(fh)

    total_docs: int = len(docs)
    all_chunks: list[dict] = []

    for doc in docs:
        section_chunks = chunk_by_section(
            text=doc["text"],
            source=doc["source"],
            law_type=doc["law_type"],
        )
        all_chunks.extend(section_chunks)

    if all_chunks:
        chunk_lengths = [len(c["text"]) for c in all_chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
    else:
        avg_length = min_length = max_length = 0

    logger.info(
        "Domain '%s': %d doc(s) → %d chunk(s) | avg_len=%.0f, min_len=%d, max_len=%d",
        domain,
        total_docs,
        len(all_chunks),
        avg_length,
        min_length,
        max_length,
    )

    output_path: Path = Path(processed_dir) / f"{domain}_chunks.json"
    with output_path.open(mode="w", encoding=ENCODING) as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    logger.info("Saved %d chunks to %s", len(all_chunks), output_path)
    return all_chunks


def main() -> None:
    """Chunk all domains and print a summary table."""
    summary_rows: list[dict] = []

    for domain in DOMAINS:
        try:
            chunks = chunk_domain(domain, PROCESSED_BASE_DIR)
        except FileNotFoundError as exc:
            logger.error("%s — skipping domain.", exc)
            continue

        lengths = [len(c["text"]) for c in chunks]
        summary_rows.append({
            "domain": domain,
            "docs": len({c["source"] for c in chunks}),
            "chunks": len(chunks),
            "avg_len": int(sum(lengths) / len(lengths)) if lengths else 0,
            "min_len": min(lengths) if lengths else 0,
            "max_len": max(lengths) if lengths else 0,
        })

    if not summary_rows:
        print("No domains processed. Check that ingest.py has been run first.")
        return

    header = (
        f"{'domain':12} | {'docs':>4} | {'chunks':>6} | "
        f"{'avg_len':>7} | {'min_len':>7} | {'max_len':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r['domain']:12} | {r['docs']:4} | {r['chunks']:6} | "
            f"{r['avg_len']:7} | {r['min_len']:7} | {r['max_len']:7}"
        )


if __name__ == "__main__":
    main()