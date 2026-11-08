"""RAG chain module for the compliance-copilot legal assistant.

Provides three LangChain-powered chains for querying legal documents:
  - ask: standard retrieval-augmented Q&A with citations
  - compare_laws: side-by-side comparison of two law domains
  - explain_simple: plain-English rewrite of a technical legal answer

All functions rely on a persisted Chroma vector store built by embed_store.py.
"""

from __future__ import annotations

import functools
import logging
from typing import TypedDict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import CFG
from src.embed_store import load_vectorstore

logger: logging.Logger = logging.getLogger("compliance-copilot.rag_chain")

VALID_LAW_TYPES: frozenset[str] = frozenset(CFG["domains"])
if CFG["reranker"] != "none":
    raise NotImplementedError(f"reranker={CFG['reranker']!r}: only 'none' is implemented")


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------


class Citation(TypedDict):
    """A single citation extracted from a retrieved source document.

    Attributes:
        source: The filename or document identifier of the source.
        section: The section heading or identifier within the source document.
    """

    source: str
    section: str


class AskResult(TypedDict):
    """The structured result returned by ask() and explain_simple().

    Attributes:
        answer: The generated answer text.
        citations: A list of citation mappings extracted from source documents.
    """

    answer: str
    citations: list[Citation]


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Cached ChatOllama. num_ctx is explicit: Ollama's default silently truncated long prompts."""
    logger.info("Initialising ChatOllama model=%s num_ctx=%s", CFG["llm_model"], CFG["num_ctx"])
    return ChatOllama(model=CFG["llm_model"], temperature=CFG["temperature"], num_ctx=CFG["num_ctx"],
                      base_url=CFG["ollama_base_url"])


@functools.lru_cache(maxsize=1)
def get_db() -> Chroma:
    """Cached handle on the configured Chroma collection."""
    return load_vectorstore()


def retrieve(question: str, law_type: str | None = None, k: int | None = None) -> list[Document]:
    """Top-k chunks for a question, optionally filtered to one law domain."""
    filt = {"law_type": law_type} if law_type else None
    return get_db().similarity_search(question, k=k or CFG["top_k"], filter=filt)


def build_prompt(question: str, docs: list[Document]) -> str:
    """Same 'stuff' layout RetrievalQA used: chunks joined by blank lines."""
    return CFG["prompt_template"].format(context="\n\n".join(d.page_content for d in docs), question=question)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_law_type(law_type: str | None, *, param_name: str = "law_type") -> None:
    """Validates that a law_type value is a member of VALID_LAW_TYPES or None.

    Args:
        law_type: The law domain string to validate, or ``None`` for all domains.
        param_name: Name of the parameter used in the error message.

    Raises:
        ValueError: If ``law_type`` is not ``None`` and not in ``VALID_LAW_TYPES``.
    """
    if law_type is not None and law_type not in VALID_LAW_TYPES:
        raise ValueError(
            f"Invalid {param_name} '{law_type}'. "
            f"Must be one of {sorted(VALID_LAW_TYPES)} or None."
        )


def _extract_citations(source_documents: list) -> list[Citation]:
    """Extracts citation metadata from a list of LangChain Document objects.

    Deduplicates citations by (source, section) pair so the same passage
    referenced multiple times only appears once in the result.

    Args:
        source_documents: A list of ``langchain_core.documents.Document``
            objects that were placed in the prompt.

    Returns:
        list[Citation]: Deduplicated citations ordered by first occurrence.
    """
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []

    for doc in source_documents:
        meta = doc.metadata or {}
        source: str = meta.get("source", "unknown")
        section: str = meta.get("section", "unknown")
        key: tuple[str, str] = (source, section)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(source=source, section=section))

    return citations


# ---------------------------------------------------------------------------
# Public chain functions
# ---------------------------------------------------------------------------


def ask(question: str, law_type: str | None = None) -> AskResult:
    """Answers a legal compliance question using retrieval-augmented generation.

    Retrieves the top ``CFG['top_k']`` chunks, optionally filtered to a single
    law domain, stuffs them into ``CFG['prompt_template']`` and asks the LLM.
    Citations are exactly the chunks that were put in the prompt.

    Args:
        question: The natural-language compliance question to answer.
        law_type: Optional domain filter. Must be one of ``VALID_LAW_TYPES``
            (``"income_tax"``, ``"gst"``, ``"rbi"``) or ``None`` to search
            across all domains.

    Returns:
        AskResult: A typed dict containing:
            - ``answer``: The LLM-generated answer string.
            - ``citations``: A list of ``Citation`` dicts with ``source`` and
              ``section`` fields extracted from the retrieved documents.

    Raises:
        ValueError: If ``law_type`` is not a valid domain or ``None``.
    """
    _validate_law_type(law_type)

    logger.info(
        "ask() called — question=%r  law_type=%s", question[:80], law_type
    )

    source_docs = retrieve(question, law_type)
    answer: str = get_llm().invoke([HumanMessage(content=build_prompt(question, source_docs))]).content
    citations: list[Citation] = _extract_citations(source_docs)

    logger.info(
        "ask() completed — answer_length=%d  citations=%d",
        len(answer),
        len(citations),
    )
    return AskResult(answer=answer, citations=citations)


def compare_laws(question: str, law_type_a: str, law_type_b: str) -> str:
    """Compares two law domains on a given topic using retrieved context.

    Independently retrieves ``CFG["compare_k"]`` documents from each of the two
    specified law domains, then asks the LLM to produce a structured
    comparison using *only* the provided context — avoiding hallucination.

    Args:
        question: The legal topic or aspect to compare across the two domains.
        law_type_a: The first law domain (must be in ``VALID_LAW_TYPES``).
        law_type_b: The second law domain (must be in ``VALID_LAW_TYPES``).

    Returns:
        str: The LLM's structured comparison text.

    Raises:
        ValueError: If either ``law_type_a`` or ``law_type_b`` is not a valid
            domain identifier.
    """
    _validate_law_type(law_type_a, param_name="law_type_a")
    _validate_law_type(law_type_b, param_name="law_type_b")

    logger.info(
        "compare_laws() called — question=%r  a=%s  b=%s",
        question[:80],
        law_type_a,
        law_type_b,
    )

    db: Chroma = get_db()
    llm: ChatOllama = get_llm()

    # Retrieve documents separately for each domain
    def _retrieve(law_type: str) -> list:
        results = db.similarity_search(
            question,
            k=CFG["compare_k"],
            filter={"law_type": law_type},
        )
        logger.debug("Retrieved %d docs for domain '%s'", len(results), law_type)
        return results

    docs_a = _retrieve(law_type_a)
    docs_b = _retrieve(law_type_b)

    def _format_docs(docs: list, label: str) -> str:
        if not docs:
            return f"[No relevant passages found for {label}]"
        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            section = meta.get("section", "unknown section")
            parts.append(
                f"  [{i}] Section: {section}\n"
                f"      {doc.page_content.strip()}"
            )
        return "\n\n".join(parts)

    context_a = _format_docs(docs_a, law_type_a.upper())
    context_b = _format_docs(docs_b, law_type_b.upper())

    prompt = (
        f"You are a senior legal analyst specialising in Indian compliance law.\n"
        f"Compare and contrast the following topic across two law domains.\n"
        f"Use ONLY the context excerpts provided below — do not draw on outside knowledge.\n"
        f"Structure your response with two clearly labelled sections.\n"
        f"Do not repeat yourself. Output the comparison exactly once.\n"
        f"TOPIC: {question}\n\n"
        f"{'=' * 60}\n"
        f"CONTEXT — {law_type_a.upper().replace('_', ' ')}\n"
        f"{'=' * 60}\n"
        f"{context_a}\n\n"
        f"{'=' * 60}\n"
        f"CONTEXT — {law_type_b.upper().replace('_', ' ')}\n"
        f"{'=' * 60}\n"
        f"{context_b}\n\n"
        f"{'=' * 60}\n"
        f"COMPARISON\n"
        f"{'=' * 60}\n"
        f"Provide a structured comparison below, using only the context above.\n"
        f"Highlight key similarities and differences between "
        f"{law_type_a.upper().replace('_', ' ')} and "
        f"{law_type_b.upper().replace('_', ' ')} with respect to the topic."
    )

    logger.debug("Sending comparison prompt to LLM...")
    response = llm.invoke([HumanMessage(content=prompt)])
    comparison: str = response.content

    logger.info(
        "compare_laws() completed — response_length=%d", len(comparison)
    )
    return comparison


def explain_simple(question: str, law_type: str | None = None) -> AskResult:
    """Returns a plain-English explanation of a legal compliance answer.

    Calls ``ask()`` to retrieve a grounded technical answer, then issues a
    second LLM prompt to rewrite that answer in simple language suitable for
    a non-lawyer audience. The original citations are preserved unchanged.

    Args:
        question: The legal compliance question to answer and simplify.
        law_type: Optional domain filter forwarded to ``ask()``. Must be one
            of ``VALID_LAW_TYPES`` or ``None``.

    Returns:
        AskResult: A typed dict containing:
            - ``answer``: The simplified, plain-English answer.
            - ``citations``: The original citations from the ``ask()`` call,
              unmodified.

    Raises:
        ValueError: If ``law_type`` is not a valid domain or ``None``.
    """
    # Validation is delegated to ask() — no need to duplicate here.
    logger.info(
        "explain_simple() called — question=%r  law_type=%s",
        question[:80],
        law_type,
    )

    technical_result: AskResult = ask(question, law_type=law_type)
    technical_answer: str = technical_result["answer"]
    citations: list[Citation] = technical_result["citations"]

    logger.debug(
        "Technical answer obtained (%d chars). Requesting plain-English rewrite...",
        len(technical_answer),
    )

    llm: ChatOllama = get_llm()

    simplify_prompt = (
        "You are a helpful assistant explaining Indian law to ordinary citizens.\n"
        "Rewrite the following legal answer in plain, simple English.\n"
        "Use short sentences, avoid jargon, and aim for a reading level of a "
        "high-school student.\n"
        "Do NOT add any new information not present in the original answer.\n\n"
        f"ORIGINAL ANSWER:\n{technical_answer}\n\n"
        "PLAIN-ENGLISH VERSION:"
    )

    response = llm.invoke([HumanMessage(content=simplify_prompt)])
    simple_answer: str = response.content

    logger.info(
        "explain_simple() completed — simplified_length=%d  citations=%d",
        len(simple_answer),
        len(citations),
    )
    return AskResult(answer=simple_answer, citations=citations)


# ---------------------------------------------------------------------------
# main — smoke-test all three chains
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs one end-to-end test for each of the three RAG chain functions.

    Logs all results at INFO level so the output is visible without enabling
    DEBUG. Designed as a quick sanity-check after building the vector store.
    """
    separator = "=" * 70

    # ------------------------------------------------------------------
    # Test 1 — ask()
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("TEST 1 — ask()  [law_type='income_tax']")
    print(separator)

    ask_result = ask(
        question="What is the tax rate for capital gains on equity mutual funds?",
        law_type="income_tax",
    )
    logger.info("[ask] answer: %s", ask_result["answer"])
    logger.info("[ask] citations (%d):", len(ask_result["citations"]))
    for c in ask_result["citations"]:
        logger.info("  source=%-30s  section=%s", c["source"], c["section"])

    print(f"ANSWER:\n{ask_result['answer']}")
    print(f"\nCITATIONS ({len(ask_result['citations'])}):")
    for c in ask_result["citations"]:
        print(f"  • {c['source']}  §{c['section']}")

    # ------------------------------------------------------------------
    # Test 2 — compare_laws()
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("TEST 2 — compare_laws()  [income_tax vs gst]")
    print(separator)

    comparison = compare_laws(
        question="How is the threshold limit for registration defined and enforced?",
        law_type_a="income_tax",
        law_type_b="gst",
    )
    logger.info("[compare_laws] response:\n%s", comparison)
    print(comparison)

    # ------------------------------------------------------------------
    # Test 3 — explain_simple()
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("TEST 3 — explain_simple()  [law_type='gst']")
    print(separator)

    simple_result = explain_simple(
        question="What are the penalties for late filing of GST returns?",
        law_type="gst",
    )
    logger.info("[explain_simple] simplified answer: %s", simple_result["answer"])
    logger.info(
        "[explain_simple] citations (%d):", len(simple_result["citations"])
    )
    for c in simple_result["citations"]:
        logger.info("  source=%-30s  section=%s", c["source"], c["section"])

    print(f"SIMPLIFIED ANSWER:\n{simple_result['answer']}")
    print(f"\nCITATIONS ({len(simple_result['citations'])}):")
    for c in simple_result["citations"]:
        print(f"  • {c['source']}  §{c['section']}")

    print(f"\n{separator}")
    print("All three chain tests completed successfully.")
    print(separator)


if __name__ == "__main__":
    main()
