"""Vector store generation pipeline for compliance-copilot.

Loads chunked legal texts, computes embeddings, and persists them into a local Chroma vector database.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import CHROMA_PERSIST_DIR

# Setup module-level logger
logger: logging.Logger = logging.getLogger("compliance-copilot.embed_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE: int = 100  # embed in batches to avoid memory spikes
ENCODING: str = "utf-8"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached HuggingFaceEmbeddings instance.

    Returns:
        HuggingFaceEmbeddings: The cached embedding model instance.
    """
    logger.info("Initializing HuggingFaceEmbeddings model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_all_chunks(processed_dir: str) -> list[dict]:
    """Loads income_tax_chunks.json, gst_chunks.json, and rbi_chunks.json and merges them.

    Args:
        processed_dir: Path to the directory containing processed chunk files.

    Returns:
        list[dict]: A merged list of all chunk dictionaries.

    Raises:
        FileNotFoundError: If any of the expected chunk files are missing.
        IOError: If there is an error reading the chunk files.
    """
    logger.info("Loading chunk files from directory: %s", processed_dir)
    domains: tuple[str, ...] = ("income_tax", "gst", "rbi")
    all_chunks: list[dict] = []

    for domain in domains:
        file_path: Path = Path(processed_dir) / f"{domain}_chunks.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected chunk file not found: {file_path}. Run chunking step first."
            )
        try:
            with file_path.open(mode="r", encoding=ENCODING) as fh:
                domain_chunks = json.load(fh)
            logger.info("Loaded %d chunks for domain '%s'", len(domain_chunks), domain)
            all_chunks.extend(domain_chunks)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from file %s: %s", file_path, exc)
            raise IOError(f"Malformed JSON file: {file_path}") from exc
        except Exception as exc:
            logger.error("Error reading file %s: %s", file_path, exc)
            raise IOError(f"Could not read file: {file_path}") from exc

    logger.info("Successfully loaded a total of %d chunks.", len(all_chunks))
    return all_chunks


def build_vectorstore(processed_dir: str, persist_dir: str) -> Chroma:
    """Loads all chunks, converts them to Documents, and indexes them in Chroma.

    Args:
        processed_dir: Path to the directory containing processed chunks.
        persist_dir: Path to the directory where the Chroma DB will be persisted.

    Returns:
        Chroma: The initialized and populated Chroma vector store instance.

    Raises:
        Exception: If any error occurs during database creation or indexing.
    """
    try:
        # Clear existing database directory to prevent duplicates and stale files
        if os.path.exists(persist_dir):
            logger.info("Clearing existing vector store directory: %s", persist_dir)
            shutil.rmtree(persist_dir)
    except Exception as exc:
        logger.warning("Could not clear directory %s: %s. Proceeding anyway...", persist_dir, exc)

    # Initialize model
    embeddings: HuggingFaceEmbeddings = get_embeddings()

    # Load chunks and construct Langchain Document objects
    chunks: list[dict] = load_all_chunks(processed_dir)
    documents: list[Document] = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "section": chunk.get("section", ""),
                "source": chunk.get("source", ""),
                "law_type": chunk.get("law_type", ""),
            }
        )
        documents.append(doc)

    logger.info("Initializing Chroma vector store at %s", persist_dir)
    try:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

        total_docs: int = len(documents)
        logger.info("Indexing %d documents in batches of %d...", total_docs, BATCH_SIZE)

        for i in range(0, total_docs, BATCH_SIZE):
            batch: list[Document] = documents[i:i + BATCH_SIZE]
            db.add_documents(batch)
            logger.info(
                "Indexed batch %d/%d (documents %d to %d)",
                (i // BATCH_SIZE) + 1,
                (total_docs - 1) // BATCH_SIZE + 1,
                i + 1,
                min(i + BATCH_SIZE, total_docs)
            )

        logger.info("Persisting Chroma database to disk...")
        db.persist()
        logger.info("Vector store successfully built and persisted.")
        return db

    except Exception as exc:
        logger.error("Error occurred while building the Chroma vector store: %s", exc)
        raise exc


def load_vectorstore(persist_dir: str) -> Chroma:
    """Loads an existing persisted Chroma DB from the disk without re-embedding.

    Args:
        persist_dir: Path to the directory where the Chroma DB is persisted.

    Returns:
        Chroma: The loaded Chroma vector store instance.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Chroma persist directory not found at: {persist_dir}. Please build the vector store first."
        )

    logger.info("Loading existing Chroma vector store from: %s", persist_dir)
    embeddings: HuggingFaceEmbeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


def main() -> None:
    """Main execution function to build the vector store pipeline."""
    # Find paths relative to project root
    project_root: Path = Path(__file__).resolve().parent.parent
    processed_dir: str = str(project_root / "data" / "processed")
    persist_dir: str = CHROMA_PERSIST_DIR

    start_time: float = time.time()
    logger.info("Starting compliance-copilot vector store pipeline...")

    try:
        # Build the vectorstore and index the documents
        db = build_vectorstore(processed_dir, persist_dir)

        # Log total documents indexed and time taken
        doc_count: int = db._collection.count()
        elapsed_time: float = time.time() - start_time
        logger.info(
            "Vector store pipeline completed. Total documents indexed: %d | Time taken: %.2f seconds",
            doc_count,
            elapsed_time
        )

    except Exception as exc:
        logger.error("Vector store pipeline execution failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
