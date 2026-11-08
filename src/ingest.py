"""Markdown ingestion pipeline for the compliance-copilot application.

Reads raw Markdown (.md) law documents from ``data/raw/{domain}/``,
applies text-cleaning heuristics, and writes structured JSON output to
``data/processed/{domain}.json``.

Each output JSON file is a list of dicts with the schema::

    {
        "source":   "<filename>",
        "law_type": "<domain>",
        "text":     "<cleaned text>"
    }

Usage::

    python -m src.ingest

Or import and call :func:`ingest_domain` / :func:`main` programmatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from src.config import CFG, path

logger: logging.Logger = logging.getLogger("compliance-copilot.ingest")

MIN_CONTENT_LENGTH: int = 100  # skip files shorter than this after cleaning
ENCODING: str = "utf-8"
MD_EXTENSION: str = ".md"


# ---------------------------------------------------------------------------
# Text-cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    # Only join lines that are clearly mid-sentence
    # NEVER join lines that start with # (Markdown headings)
    # The (?m)^(?!#) guard stops a heading line from swallowing the body line after it.
    text = re.sub(r'(?m)^(?!#)(.*[^.\n#])\n(?![\n#])', r'\1 ', text)
    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    # Remove repeated short lines (page headers/footers), but keep numbered headings like "## 1." that
    # legitimately repeat across annexes; dropping them merged whole sections together.
    lines = text.split('\n')
    from collections import Counter
    counts = Counter(l.strip() for l in lines if len(l.strip()) < 40)
    lines = [l for l in lines if counts.get(l.strip(), 0) <= 3 or len(l.strip()) >= 40
             or re.match(r'#+ (Para )?\d', l)]
    text = '\n'.join(lines)
    # Normalize blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_markdown_file(filepath: str) -> str:
    """Read and clean a single Markdown file.

    Args:
        filepath: Absolute or relative path to the ``.md`` file.

    Returns:
        Cleaned text content of the file.

    Raises:
        FileNotFoundError: If *filepath* does not exist on disk, with a
            descriptive message including the attempted path.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Markdown file not found at expected path: '{filepath}'. "
            "Please ensure the file exists before calling read_markdown_file()."
        )

    raw_content: str = path.read_text(encoding=ENCODING)
    cleaned: str = clean_text(raw_content)

    logger.debug(
        "Read '%s' → %d characters after cleaning.",
        path.name,
        len(cleaned),
    )
    return cleaned


# ---------------------------------------------------------------------------
# Domain ingestion
# ---------------------------------------------------------------------------

def ingest_domain(domain: str, raw_dir: str, output_dir: str) -> list[dict]:
    """Ingest all Markdown files for a single law domain.

    Reads every ``.md`` file found in *raw_dir*, cleans the text, and
    assembles a list of record dicts.  Files whose cleaned text is shorter
    than :data:`MIN_CONTENT_LENGTH` characters are skipped with a warning.
    The resulting list is serialised to ``{output_dir}/{domain}.json``.

    Args:
        domain: Law domain identifier, e.g. ``"income_tax"``.
        raw_dir: Directory containing raw ``.md`` source files for *domain*.
        output_dir: Directory where the ``{domain}.json`` output will be
            written.  Created automatically if it does not exist.

    Returns:
        List of record dicts, each with keys ``"source"``, ``"law_type"``,
        and ``"text"``.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    # Collect candidate files
    md_files: list[Path] = sorted(p for p in raw_path.glob(f"*{MD_EXTENSION}") if p.name not in CFG["exclude_sources"])
    total_attempted: int = len(md_files)

    logger.info(
        "Domain '%s': found %d .md file(s) in '%s'.",
        domain,
        total_attempted,
        raw_dir,
    )

    records: list[dict] = []

    for md_file in md_files:
        try:
            cleaned_text: str = read_markdown_file(str(md_file))
        except FileNotFoundError as exc:
            logger.error(
                "Skipping '%s' — file not found during ingestion: %s",
                md_file.name,
                exc,
            )
            continue
        except OSError as exc:
            logger.error(
                "Skipping '%s' — OS error while reading: %s",
                md_file.name,
                exc,
            )
            continue

        if len(cleaned_text) < MIN_CONTENT_LENGTH:
            logger.warning(
                "Skipping '%s' — cleaned text too short (%d chars < minimum %d).",
                md_file.name,
                len(cleaned_text),
                MIN_CONTENT_LENGTH,
            )
            continue

        records.append(
            {
                "source": md_file.name,
                "law_type": domain,
                "text": cleaned_text,
            }
        )

    # Persist output
    output_path.mkdir(parents=True, exist_ok=True)
    output_file: Path = output_path / f"{domain}.json"
    with output_file.open(mode="w", encoding=ENCODING) as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    successfully_ingested: int = len(records)
    logger.info(
        "Domain '%s': ingested %d / %d file(s) → '%s'.",
        domain,
        successfully_ingested,
        total_attempted,
        output_file,
    )

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the ingestion pipeline for all three law domains.

    Iterates over ``CFG['domains']``, calls :func:`ingest_domain` for each,
    then prints a formatted summary table to *stdout*::

        domain        | files_found | files_ingested | total_chars
        ------------- | ----------- | -------------- | -----------
        income_tax    |          42 |             40 |     2345678
        gst           |          31 |             30 |     1987654
        rbi           |          28 |             27 |     1654321
    """
    summary_rows: list[dict] = []

    for domain in CFG["domains"]:
        raw_dir: str = str(path("raw_dir") / domain)
        records: list[dict] = ingest_domain(
            domain=domain,
            raw_dir=raw_dir,
            output_dir=str(path("processed_dir")),
        )

        # Count .md files found (re-check so summary is accurate even if
        # ingest_domain encountered errors mid-loop).
        raw_path = Path(raw_dir)
        files_found: int = len(list(raw_path.glob(f"*{MD_EXTENSION}"))) if raw_path.exists() else 0
        files_ingested: int = len(records)
        total_chars: int = sum(len(r["text"]) for r in records)

        summary_rows.append(
            {
                "domain": domain,
                "files_found": files_found,
                "files_ingested": files_ingested,
                "total_chars": total_chars,
            }
        )

    # --- Print summary table ---
    col_domain: int = max(len("domain"), *(len(r["domain"]) for r in summary_rows))
    col_found: int = max(len("files_found"), *(len(str(r["files_found"])) for r in summary_rows))
    col_ingested: int = max(len("files_ingested"), *(len(str(r["files_ingested"])) for r in summary_rows))
    col_chars: int = max(len("total_chars"), *(len(str(r["total_chars"])) for r in summary_rows))

    header: str = (
        f"{'domain':<{col_domain}} | "
        f"{'files_found':>{col_found}} | "
        f"{'files_ingested':>{col_ingested}} | "
        f"{'total_chars':>{col_chars}}"
    )
    separator: str = (
        f"{'-' * col_domain}-+-"
        f"{'-' * col_found}-+-"
        f"{'-' * col_ingested}-+-"
        f"{'-' * col_chars}"
    )

    print(header)
    print(separator)
    for row in summary_rows:
        print(
            f"{row['domain']:<{col_domain}} | "
            f"{row['files_found']:>{col_found}} | "
            f"{row['files_ingested']:>{col_ingested}} | "
            f"{row['total_chars']:>{col_chars}}"
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
