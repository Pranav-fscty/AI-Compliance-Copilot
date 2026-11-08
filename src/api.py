import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import CFG
from src.rag_chain import ask, compare_laws, explain_simple, get_db, get_llm

# Configure logging for the api module
logger: logging.Logger = logging.getLogger("compliance-copilot.api")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request model for /ask and /simplify endpoints."""
    question: str = Field(..., description="The natural-language compliance question to ask.")
    law_type: str | None = Field(default=None, description="Optional law domain filter (e.g., 'income_tax', 'gst', 'rbi').")


class CompareRequest(BaseModel):
    """Request model for /compare endpoint."""
    question: str = Field(..., description="The legal topic or aspect to compare across the two domains.")
    law_type_a: str = Field(..., description="The first law domain (e.g., 'income_tax', 'gst', 'rbi').")
    law_type_b: str = Field(..., description="The second law domain (e.g., 'income_tax', 'gst', 'rbi').")


class CitationModel(BaseModel):
    """Model representing a source document citation."""
    source: str = Field(..., description="The filename or document identifier of the source.")
    section: str = Field(..., description="The section heading or identifier within the source document.")


class AskResponse(BaseModel):
    """Response model for /ask and /simplify endpoints."""
    answer: str = Field(..., description="The generated answer from the assistant.")
    citations: list[CitationModel] = Field(..., description="A list of citation mappings extracted from source documents.")


class CompareResponse(BaseModel):
    """Response model for /compare endpoint."""
    comparison: str = Field(..., description="The generated comparison text.")


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    logger.info("Initializing application caches at startup...")
    try:
        get_db()
        get_llm()
    except Exception as e:  # stay up so /health can report 503 instead of the container crash-looping
        logger.exception("Failed to warm up database or LLM caches during startup: %s", str(e))
    logger.info("Application startup complete")
    yield


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="AI Compliance Copilot",
    description="RAG-based assistant for Indian regulatory law",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def handle_ask(request: AskRequest) -> AskResponse:
    """Answers a legal compliance question using retrieval-augmented generation."""
    logger.info("POST /ask received: question=%r, law_type=%s", request.question[:80], request.law_type)
    try:
        result = ask(question=request.question, law_type=request.law_type)
        return AskResponse(
            answer=result["answer"],
            citations=[CitationModel(source=c["source"], section=c["section"]) for c in result["citations"]]
        )
    except ValueError as e:
        logger.warning("Validation error in /ask: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unhandled error in /ask endpoint: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/compare", response_model=CompareResponse)
def handle_compare(request: CompareRequest) -> CompareResponse:
    """Compares two law domains on a given topic using retrieved context."""
    logger.info("POST /compare received: question=%r, law_type_a=%s, law_type_b=%s",
                request.question[:80], request.law_type_a, request.law_type_b)
    try:
        comparison_result = compare_laws(
            question=request.question,
            law_type_a=request.law_type_a,
            law_type_b=request.law_type_b
        )
        return CompareResponse(comparison=comparison_result)
    except ValueError as e:
        logger.warning("Validation error in /compare: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unhandled error in /compare endpoint: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/simplify", response_model=AskResponse)
def handle_simplify(request: AskRequest) -> AskResponse:
    """Returns a simplified plain-English explanation of a legal compliance answer."""
    logger.info("POST /simplify received: question=%r, law_type=%s", request.question[:80], request.law_type)
    try:
        result = explain_simple(question=request.question, law_type=request.law_type)
        return AskResponse(
            answer=result["answer"],
            citations=[CitationModel(source=c["source"], section=c["section"]) for c in result["citations"]]
        )
    except ValueError as e:
        logger.warning("Validation error in /simplify: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unhandled error in /simplify endpoint: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/health")
def handle_health() -> JSONResponse:
    """200 if the vector index is readable and non-empty and the LLM server answers; 503 otherwise."""
    checks: dict[str, str] = {}
    try:
        checks["index"] = "ok" if get_db()._collection.count() > 0 else "empty"
    except Exception:  # noqa: BLE001 - any failure means unhealthy; details go to the log only
        logger.exception("health: index check failed")
        checks["index"] = "error"
    try:
        httpx.get(f"{CFG['ollama_base_url']}/api/version", timeout=2).raise_for_status()
        checks["llm"] = "ok"
    except Exception:  # noqa: BLE001
        logger.exception("health: llm check failed")
        checks["llm"] = "error"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"status": "ok" if ok else "degraded", **checks}, status_code=200 if ok else 503)
