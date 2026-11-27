"""FastAPI application for MWIZERWA HTTP API."""

from __future__ import annotations

import base64
import json
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
    MemoryConsentRequest,
    MemoryConsentResponse,
    TranscribeResponse,
    VoiceResponse,
    TurnContextRequest,
)
from src.memory.consent import ConsentDecision
from src.memory.dev_consent import dev_memory_consent_enabled
from src.api.context_mapping import build_turn_context
from src.conversation.context_integration import TurnContextValidationError
from src.api.readiness import build_backend_readiness
from src.api.service import _public_metadata, get_api_service
from src.asr.exceptions import UnsupportedASRLanguageError, WhisperASRNotAvailableError
from src.tts.tts_engine import TTSSynthesisError, UnsupportedTTSLanguageError

logger = logging.getLogger(__name__)

SUPPORTED_TTS_LANGS = ("rw", "en", "fr")
SUPPORTED_ASR_LANGS = ("rw", "en", "fr")


def _normalize_asr_language_form(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    lang = language.strip().lower()
    if not lang:
        return None
    if lang not in SUPPORTED_ASR_LANGS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language '{language}'. Supported: rw, en, fr.",
        )
    return lang


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
    backends = build_backend_readiness(service)
    if service.engine_initialized:
        return HealthResponse(
            status="ready",
            engine_initialized=True,
            backends=backends,
        )
    return HealthResponse(
        status="unavailable",
        engine_initialized=False,
        backends=backends,
    )


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
        turn_context = build_turn_context(body.context)
        session_id, result = service.process_chat(
            user_text=body.user_text,
            session_id=body.session_id,
            mode=body.mode,
            language=body.language,
            user_id=body.user_id,
            context=turn_context,
        )
    except TurnContextValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
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
        recommended_exercise_id=result.recommended_exercise_id,
        metadata=_public_metadata(result.metadata),
    )


@app.post(
    "/memory/consent",
    response_model=MemoryConsentResponse,
    summary="Mock memory consent (development — application backend replaces this)",
)
def memory_consent(body: MemoryConsentRequest):
    """
    Simulated user consent for persistent memory proposals.
    Production: the Mwizerwa app records consent after UI interaction and calls a trusted internal API.
    """
    if not dev_memory_consent_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory consent endpoint is disabled in this environment.",
        )
    service = get_api_service()
    decision = ConsentDecision(body.decision)
    summary = service.memory_coordinator.apply_consent_decision(
        body.user_id,
        body.proposal_id,
        decision,
    )
    applied = decision == ConsentDecision.ACCEPTED and summary is not None
    return MemoryConsentResponse(
        user_id=body.user_id,
        proposal_id=body.proposal_id,
        applied=applied or decision == ConsentDecision.DECLINED,
        summary_updated=summary is not None,
        memory_summary=summary,
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
):
    text = text.strip()
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="text must not be empty.")

    service = get_api_service()
    wav_path: Optional[Path] = None
    try:
        wav_path = service.synthesize_tts(text, language=language)
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
    language: Optional[Literal["rw", "en", "fr"]] = Form(
        default=None,
        description="ASR language. Omit for Kinyarwanda (default).",
    ),
):
    service = get_api_service()
    upload_path: Optional[Path] = None
    try:
        asr_lang = _normalize_asr_language_form(language)
        upload_path = write_upload_to_tempfile(audio)
        transcription, effective_lang = service.transcribe_file(
            upload_path, language=asr_lang
        )
        return TranscribeResponse(
            transcription=transcription,
            language=effective_lang,
        )
    except HTTPException:
        raise
    except UnsupportedASRLanguageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except WhisperASRNotAvailableError as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
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
    language: Optional[Literal["rw", "en", "fr"]] = Form(
        default=None,
        description="ASR and conversation language hint. Omit for Kinyarwanda (default).",
    ),
    context_json: Optional[str] = Form(
        default=None,
        description="Optional JSON object matching TurnContextRequest fields.",
    ),
):
    service = get_api_service()
    if not service.engine_initialized:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant is not ready.",
        )

    upload_path: Optional[Path] = None
    turn_context = None
    if context_json is not None and context_json.strip():
        try:
            turn_context = build_turn_context(
                TurnContextRequest.model_validate(json.loads(context_json))
            )
        except TurnContextValidationError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        except json.JSONDecodeError as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="context_json must be valid JSON.",
            ) from e
        except Exception as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid context_json: {e}",
            ) from e
    try:
        asr_lang = _normalize_asr_language_form(language)
        upload_path = write_upload_to_tempfile(audio)
        sid, transcript, used_asr_lang, result, audio_bytes, audio_error = (
            service.process_voice_turn(
                upload_path,
                session_id=session_id,
                mode=mode,
                language=asr_lang,
                context=turn_context,
            )
        )
        tts_ok = audio_bytes is not None
        return VoiceResponse(
            session_id=sid,
            transcription=transcript,
            response_text=result.response_text,
            mode=result.mode.value,
            language=result.language,
            asr_language=used_asr_lang,
            recommended_exercise_id=result.recommended_exercise_id,
            metadata=_public_metadata(result.metadata),
            audio_base64=(
                base64.b64encode(audio_bytes).decode("ascii") if tts_ok else None
            ),
            audio_status="success" if tts_ok else "failed",
            audio_error=audio_error,
            audio_media_type="audio/wav" if tts_ok else None,
        )
    except HTTPException:
        raise
    except UnsupportedASRLanguageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except WhisperASRNotAvailableError as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except TurnContextValidationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
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