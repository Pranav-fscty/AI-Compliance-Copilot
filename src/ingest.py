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
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Module-level logger (independent of src.config so we never trigger the
# OPENAI_API_KEY validation just by running the ingestion pipeline).
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger("compliance-copilot.ingest")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CONTENT_LENGTH: int = 100          # skip files shorter than this after cleaning
ENCODING: str = "utf-8"

# Supported law domains
DOMAINS: tuple[str, ...] = ("income_tax", "gst", "rbi")

# Directory layout (resolved relative to this file's parent package root)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_BASE_DIR: str = str(_PROJECT_ROOT / "data" / "raw")
PROCESSED_BASE_DIR: str = str(_PROJECT_ROOT / "data" / "processed")

# Markdown file extension
MD_EXTENSION: str = ".md"

# Repeated-line threshold: lines appearing more than this many times are
# treated as headers/footers and removed.
REPEATED_LINE_MAX_COUNT: int = 3

# Minimum line length below which repetition-check is applied
REPEATED_LINE_LENGTH_THRESHOLD: int = 40

# Number of consecutive blank lines to collapse into one
BLANK_LINE_COLLAPSE_THRESHOLD: int = 3

# Line-ending punctuation / markers that indicate a line is "complete"
# (i.e. should *not* be joined with the next line).
_COMPLETE_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:\.\s*$|:\s*$|#\s)"
)

# Lines consisting solely of digits (page-number artifacts)
_PAGE_NUMBER_PATTERN: re.Pattern[str] = re.compile(r"^\d+\s*$")

# Three-or-more consecutive blank lines
_EXCESS_BLANK_LINES_PATTERN: re.Pattern[str] = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Text-cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    # Only join lines that are clearly mid-sentence
    # NEVER join lines that start with # (Markdown headings)
    text = re.sub(r'(?<![.\n#])\n(?![\n#])', ' ', text)
    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    # Remove repeated short lines
    lines = text.split('\n')
    from collections import Counter
    counts = Counter(l.strip() for l in lines if len(l.strip()) < 40)
    lines = [l for l in lines if counts.get(l.strip(), 0) <= 3 or len(l.strip()) >= 40]
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
    md_files: list[Path] = sorted(raw_path.glob(f"*{MD_EXTENSION}"))
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

    Iterates over :data:`DOMAINS`, calls :func:`ingest_domain` for each,
    then prints a formatted summary table to *stdout*::

        domain        | files_found | files_ingested | total_chars
        ------------- | ----------- | -------------- | -----------
        income_tax    |          42 |             40 |     2345678
        gst           |          31 |             30 |     1987654
        rbi           |          28 |             27 |     1654321
    """
    summary_rows: list[dict] = []

    for domain in DOMAINS:
        raw_dir: str = os.path.join(RAW_BASE_DIR, domain)
        records: list[dict] = ingest_domain(
            domain=domain,
            raw_dir=raw_dir,
            output_dir=PROCESSED_BASE_DIR,
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
