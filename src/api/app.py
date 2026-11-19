"""FastAPI application for MWIZERWA HTTP API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from src.api.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from src.api.service import _public_metadata, get_api_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_api_service()
    try:
        service.startup()
    except Exception:
        logger.error("Startup load failed; /health will report unavailable until restart")
    yield


app = FastAPI(
    title="MWIZERWA AI API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
)
def health():
    service = get_api_service()
    if service.engine_initialized:
        return HealthResponse(status="ready", engine_initialized=True)
    return HealthResponse(status="unavailable", engine_initialized=False)


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def chat(body: ChatRequest):
    service = get_api_service()
    if not service.engine_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant is not ready. Retry after /health reports ready.",
        )

    try:
        session_id, result = service.process_chat(
            user_text=body.user_text,
            session_id=body.session_id,
            mode=body.mode,
            language=body.language,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process this message. Please try again.",
        ) from None

    return ChatResponse(
        session_id=session_id,
        response_text=result.response_text,
        mode=result.mode.value,
        language=result.language,
        metadata=_public_metadata(result.metadata),
    )