"""Experiment tracking module for compliance-copilot.

Compares retrieval accuracy and average answer length across different chunk sizes
and embedding models, logging the parameters and metrics to MLflow.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
import mlflow

# Import the text splitter for parameter-based chunking
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import src.chunk as chunk
import src.embed_store as embed_store
import src.rag_chain as rag_chain
from src.config import CHROMA_PERSIST_DIR
from src.eval import run_eval

# Setup module logger
logger: logging.Logger = logging.getLogger("compliance-copilot.experiments")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME: str = "compliance-copilot-retrieval"
EMBEDDING_MODELS: list[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]
CHUNK_SIZES: list[int] = [300, 500, 800]
CHUNK_OVERLAP: int = 100

# Global list to store experiment results for printing in main()
_results: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_custom_chunks(
    processed_dir: str, chunk_size: int, chunk_overlap: int
) -> list[dict]:
    """Generates custom text chunks using specific chunk_size and chunk_overlap parameters.

    Reads clean processed JSON files for each domain, splits them into sections
    by Markdown headings, and then sub-chunks those sections using the
    RecursiveCharacterTextSplitter.

    Args:
        processed_dir: Path to the directory containing processed domain files.
        chunk_size: Maximum size of each text chunk (in characters).
        chunk_overlap: Overlap between consecutive text chunks (in characters).

    Returns:
        list[dict]: A list of generated chunk dictionaries containing keys:
            'text', 'section', 'source', 'law_type'.
    """
    logger.info(
        "Generating custom chunks (size=%d, overlap=%d) from processed files in: %s",
        chunk_size,
        chunk_overlap,
        processed_dir,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks: list[dict] = []

    for domain in chunk.DOMAINS:
        input_path = Path(processed_dir) / f"{domain}.json"
        if not input_path.exists():
            logger.warning(
                "Processed domain file not found, skipping: %s", input_path
            )
            continue

        try:
            with input_path.open(mode="r", encoding=chunk.ENCODING) as fh:
                docs = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load processed file %s: %s", input_path, exc)
            continue

        for doc in docs:
            # Split document text into sections based on '## ' markdown headings
            sections = re.split(chunk.HEADING_PATTERN, doc["text"])
            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue
                heading = lines[0].replace("## ", "").strip()
                body = "\n".join(lines[1:]).strip()

                if len(body) < chunk.MIN_CHUNK_CHARS:
                    continue

                # Sub-chunk the section body based on current experiment parameters
                sub_chunks = splitter.split_text(body)
                for sub_chunk in sub_chunks:
                    all_chunks.append(
                        {
                            "text": sub_chunk.strip(),
                            "section": heading if heading else "Preamble",
                            "source": doc["source"],
                            "law_type": doc["law_type"],
                        }
                    )

    logger.info(
        "Successfully generated %d custom chunks across all domains.",
        len(all_chunks),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Core Experiment Functions
# ---------------------------------------------------------------------------
def run_experiment(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    eval_questions_path: str,
) -> float:
    """Runs a single evaluation experiment and logs parameters/metrics to MLflow.

    Rebuilds the vector store using the customized chunks and embedding model,
    executes the evaluation pipeline, computes the average answer length, and
    records everything under the designated MLflow experiment.

    Args:
        embedding_model: The name or path of the HuggingFace embedding model.
        chunk_size: The chunk size parameter for text splitting.
        chunk_overlap: The chunk overlap parameter for text splitting.
        eval_questions_path: Path to the evaluation questions JSON file.

    Returns:
        float: The retrieval accuracy measured from this experiment run.
    """
    # 1. Sets up mlflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"{embedding_model.split('/')[-1]}_chunk{chunk_size}"

    # 2. Start MLflow run
    with mlflow.start_run(run_name=run_name):
        logger.info("=== Starting experiment run: %s ===", run_name)

        # Log parameters
        mlflow.log_params(
            {
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

        # Locate paths
        project_root = Path(__file__).resolve().parent.parent
        processed_dir = str(project_root / "data" / "processed")

        # Generate custom chunks with these settings
        custom_chunks = get_custom_chunks(processed_dir, chunk_size, chunk_overlap)

        # Monkey-patch load_all_chunks to return our custom split chunks
        def mock_load_all_chunks(p_dir: str) -> list[dict]:
            return custom_chunks

        original_load_all_chunks = embed_store.load_all_chunks
        embed_store.load_all_chunks = mock_load_all_chunks

        # Update embedding model names and clear cached embedding/vectorstore singletons
        original_embed_model = embed_store.EMBEDDING_MODEL_NAME
        original_rag_model = rag_chain.EMBEDDING_MODEL_NAME

        embed_store.EMBEDDING_MODEL_NAME = embedding_model
        rag_chain.EMBEDDING_MODEL_NAME = embedding_model

        embed_store.get_embeddings.cache_clear()
        rag_chain.get_db.cache_clear()

        # Wrap rag_chain.ask to intercept and log answer lengths
        original_ask = rag_chain.ask
        answer_lengths: list[int] = []

        def wrapped_ask(question: str, law_type: str | None = None):
            res = original_ask(question, law_type)
            if isinstance(res, dict) and "answer" in res and res["answer"]:
                answer_lengths.append(len(res["answer"]))
            else:
                answer_lengths.append(0)
            return res

        rag_chain.ask = wrapped_ask

        try:
            # Rebuild the vector store with these settings
            logger.info("Rebuilding vector store for run: %s", run_name)
            embed_store.build_vectorstore(processed_dir, CHROMA_PERSIST_DIR)

            # Run evaluation
            logger.info("Running evaluation for run: %s", run_name)
            eval_summary = run_eval(eval_questions_path)

            retrieval_accuracy = eval_summary.get("retrieval_accuracy", 0.0)
            avg_answer_length = (
                sum(answer_lengths) / len(answer_lengths)
                if answer_lengths
                else 0.0
            )

            # Log metrics
            mlflow.log_metric("retrieval_accuracy", retrieval_accuracy)
            mlflow.log_metric("avg_answer_length", avg_answer_length)

            logger.info(
                "Logged metrics for run %s -> retrieval_accuracy: %.4f, avg_answer_length: %.2f",
                run_name,
                retrieval_accuracy,
                avg_answer_length,
            )

            return retrieval_accuracy

        finally:
            # Restore all original functions, constants, and clear caches
            embed_store.load_all_chunks = original_load_all_chunks
            embed_store.EMBEDDING_MODEL_NAME = original_embed_model
            rag_chain.EMBEDDING_MODEL_NAME = original_rag_model
            rag_chain.ask = original_ask
            embed_store.get_embeddings.cache_clear()
            rag_chain.get_db.cache_clear()


def run_all_experiments(eval_questions_path: str) -> None:
    """Runs evaluations for all embedding model and chunk size parameter variations.

    Logs the total number of combinations and a summary table of the results.
    Each experiment run is isolated inside a try-except block so a failure
    in one run does not prevent the execution of subsequent runs.

    Args:
        eval_questions_path: Path to the evaluation questions JSON file.
    """
    total_combinations = len(EMBEDDING_MODELS) * len(CHUNK_SIZES)
    logger.info(
        "Total parameter combinations to evaluate: %d", total_combinations
    )

    # Clear previous results
    _results.clear()

    for model in EMBEDDING_MODELS:
        for size in CHUNK_SIZES:
            logger.info(
                "Starting experiment: model=%s, chunk_size=%d, chunk_overlap=%d",
                model,
                size,
                CHUNK_OVERLAP,
            )
            try:
                accuracy = run_experiment(
                    embedding_model=model,
                    chunk_size=size,
                    chunk_overlap=CHUNK_OVERLAP,
                    eval_questions_path=eval_questions_path,
                )
                _results.append(
                    {"model": model, "chunk_size": size, "accuracy": accuracy}
                )
            except Exception as exc:
                logger.error(
                    "Experiment run failed for model=%s, chunk_size=%d: %s",
                    model,
                    size,
                    exc,
                    exc_info=True,
                )
                # Record a fallback failure result to complete the table
                _results.append(
                    {"model": model, "chunk_size": size, "accuracy": 0.0}
                )

    # Log summary table via the logger
    logger.info("=" * 80)
    logger.info("EXPERIMENT RETRIEVAL ACCURACY SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-45s | %-10s | %-18s",
        "Embedding Model",
        "Chunk Size",
        "Retrieval Accuracy",
    )
    logger.info("-" * 80)
    for r in _results:
        logger.info(
            "%-45s | %-10d | %-18.4f",
            r["model"],
            r["chunk_size"],
            r["accuracy"],
        )
    logger.info("=" * 80)


def main() -> None:
    """Main entry point to execute the experiment suite and print results."""
    # Ensure at least a basic handler is present for main execution
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            stream=sys.stdout,
        )

    project_root = Path(__file__).resolve().parent.parent
    default_eval_path = str(project_root / "eval" / "questions.json")

    # Run the experiment suite
    run_all_experiments(default_eval_path)

    # Print summary table as the only allowed stdout print
    print("\n" + "=" * 80)
    print("EXPERIMENT RETRIEVAL ACCURACY SUMMARY")
    print("=" * 80)
    print(
        f"{'Embedding Model':<45} | {'Chunk Size':<10} | {'Retrieval Accuracy':<18}"
    )
    print("-" * 80)
    for r in _results:
        print(f"{r['model']:<45} | {r['chunk_size']:<10} | {r['accuracy']:<18.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
