"""Vector store: embed chunk files into a named Chroma collection, and load it for querying.

Building only replaces CFG['collection']; other collections in the same chroma_dir are untouched,
so several embedders / chunk sizes can live side by side.
"""

from __future__ import annotations

import functools
import json
import logging
import time

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import CFG, path

logger: logging.Logger = logging.getLogger("compliance-copilot.embed_store")

BATCH_SIZE: int = 100  # embed in batches to avoid memory spikes
ENCODING: str = "utf-8"


@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached embedder; normalized so cosine == dot product (MiniLM already normalizes, others may not)."""
    logger.info("Initializing embedding model: %s", CFG["embedding_model"])
    return HuggingFaceEmbeddings(model_name=CFG["embedding_model"], encode_kwargs={"normalize_embeddings": True})


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(path("chroma_dir")), settings=Settings(anonymized_telemetry=False))


def load_all_chunks() -> list[dict]:
    """Merge {domain}_chunks.json for every configured domain."""
    all_chunks: list[dict] = []
    for domain in CFG["domains"]:
        file_path = path("processed_dir") / f"{domain}_chunks.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Expected chunk file not found: {file_path}. Run src.chunk first.")
        try:
            all_chunks.extend(json.loads(file_path.read_text(encoding=ENCODING)))
        except json.JSONDecodeError as exc:
            raise IOError(f"Malformed JSON file: {file_path}") from exc
    logger.info("Loaded %d chunks.", len(all_chunks))
    return all_chunks


def build_vectorstore() -> Chroma:
    """(Re)create CFG['collection'] from the chunk files."""
    client = get_client()
    if CFG["collection"] in [c.name for c in client.list_collections()]:
        logger.info("Dropping existing collection %s", CFG["collection"])
        client.delete_collection(CFG["collection"])

    chunks = load_all_chunks()
    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk ids; re-run src.chunk")
    docs = [Document(page_content=c["text"], metadata={k: c[k] for k in ("id", "section", "source", "law_type", "chunk_index")})
            for c in chunks]

    db = Chroma(client=client, collection_name=CFG["collection"], embedding_function=get_embeddings(),
                collection_metadata={"hnsw:space": CFG["distance"], "hnsw:search_ef": CFG["hnsw_search_ef"]})
    for i in range(0, len(docs), BATCH_SIZE):
        db.add_documents(docs[i:i + BATCH_SIZE], ids=ids[i:i + BATCH_SIZE])
        logger.info("Indexed %d/%d", min(i + BATCH_SIZE, len(docs)), len(docs))
    return db


def load_vectorstore() -> Chroma:
    """Open CFG['collection'] without re-embedding. Raises if it is missing or empty."""
    client = get_client()
    if CFG["collection"] not in [c.name for c in client.list_collections()]:
        raise FileNotFoundError(f"Collection {CFG['collection']!r} not found in {path('chroma_dir')}. Run src.embed_store.")
    db = Chroma(client=client, collection_name=CFG["collection"], embedding_function=get_embeddings())
    if db._collection.count() == 0:
        raise FileNotFoundError(f"Collection {CFG['collection']!r} is empty. Run src.embed_store.")
    return db


def main() -> None:
    start = time.time()
    db = build_vectorstore()
    logger.info("Indexed %d chunks into %s in %.1fs (metadata=%s)", db._collection.count(), CFG["collection"],
                time.time() - start, db._collection.metadata)


if __name__ == "__main__":
    main()
