"""FastAPI application for MWIZERWA HTTP API."""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from src.api.audio_utils import max_upload_bytes, safe_unlink, write_upload_to_tempfile
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    TranscribeResponse,
    VoiceResponse,
)
from src.api.service import _public_metadata, get_api_service
from src.tts.tts_engine import TTSSynthesisError, UnsupportedTTSLanguageError

logger = logging.getLogger(__name__)

SUPPORTED_TTS_LANGS = ("rw", "en", "fr")


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
    version="1.1.0",
    lifespan=lifespan,
    description=(
        "Text and voice API for MWIZERWA. Languages: Kinyarwanda (rw), English (en), French (fr). "
        f"Max audio upload: {max_upload_bytes() // (1024 * 1024)} MB."
    ),
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


@app.post(
    "/tts",
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Text-to-speech",
)
def tts(
    text: str = Form(..., description="Text to synthesize"),
    language: Optional[Literal["rw", "en", "fr"]] = Form(
        default=None,
        description="Target language. Omit to detect from text (assistant-response routing).",
    ),
    speaker_id: int = Form(default=0, ge=0, le=2),
):
    text = text.strip()
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="text must not be empty.")

    service = get_api_service()
    wav_path: Optional[Path] = None
    try:
        wav_path = service.synthesize_tts(text, language=language, speaker_id=speaker_id)
        data = wav_path.read_bytes()
        safe_unlink(wav_path)
        return Response(
            content=data,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model files are not available on this server.",
        ) from e
    except (UnsupportedTTSLanguageError, TTSSynthesisError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception:
        logger.exception("TTS failed")
        safe_unlink(wav_path)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech synthesis failed.",
        ) from None


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Speech-to-text",
)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, WebM, …)"),
):
    service = get_api_service()
    upload_path: Optional[Path] = None
    try:
        upload_path = write_upload_to_tempfile(audio)
        transcription = service.transcribe_file(upload_path)
        return TranscribeResponse(transcription=transcription)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR model is not available on this server.",
        ) from e
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed.",
        ) from None
    finally:
        safe_unlink(upload_path)


@app.post(
    "/voice",
    response_model=VoiceResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Voice-to-voice (ASR → chat → TTS)",
)
async def voice(
    audio: UploadFile = File(..., description="User speech audio"),
    session_id: Optional[str] = Form(default=None),
    mode: Optional[Literal["gentle", "normal", "motivational"]] = Form(default=None),
    speaker_id: int = Form(default=0, ge=0, le=2),
):
    service = get_api_service()
    if not service.engine_initialized:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant is not ready.",
        )

    upload_path: Optional[Path] = None
    try:
        upload_path = write_upload_to_tempfile(audio)
        sid, transcript, result, audio_bytes = service.process_voice_turn(
            upload_path,
            session_id=session_id,
            mode=mode,
            speaker_id=speaker_id,
        )
        return VoiceResponse(
            session_id=sid,
            transcription=transcript,
            response_text=result.response_text,
            mode=result.mode.value,
            language=result.language,
            metadata=_public_metadata(result.metadata),
            audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
            audio_media_type="audio/wav",
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required speech or language models are not available.",
        ) from e
    except Exception:
        logger.exception("Voice turn failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice processing failed.",
        ) from None
    finally:
        safe_unlink(upload_path)