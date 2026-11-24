"""API request/response models (public contract only)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    engine_initialized: bool


SUPPORTED_CHAT_LANGUAGES = frozenset({"rw", "en", "fr"})
_MAX_SESSION_ID_LEN = 128
_MAX_USER_ID_LEN = 128


class ChatRequest(BaseModel):
    user_text: str
    session_id: Optional[str] = Field(
        default=None,
        description="Stable client session id; omit to start a new conversation.",
    )
    mode: Optional[Literal["gentle", "normal", "motivational"]] = None
    language: Optional[str] = Field(
        default=None,
        description="Optional ISO language hint (rw/en/fr).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Trusted application user id for persistent memory (server-to-server only; "
            "do not accept unauthenticated client-supplied ids in production)."
        ),
    )

    @field_validator("user_text")
    @classmethod
    def user_text_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_text must not be empty")
        return stripped

    @field_validator("language")
    @classmethod
    def language_supported(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in SUPPORTED_CHAT_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{value}'. Supported: rw, en, fr."
            )
        return normalized

    @field_validator("session_id")
    @classmethod
    def session_id_sane(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        sid = value.strip()
        if not sid:
            raise ValueError("session_id must not be empty or whitespace")
        if len(sid) > _MAX_SESSION_ID_LEN:
            raise ValueError(
                f"session_id exceeds maximum length ({_MAX_SESSION_ID_LEN})."
            )
        return sid

    @field_validator("user_id")
    @classmethod
    def user_id_sane(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        uid = value.strip()
        if not uid:
            raise ValueError("user_id must not be empty or whitespace")
        if len(uid) > _MAX_USER_ID_LEN:
            raise ValueError(f"user_id exceeds maximum length ({_MAX_USER_ID_LEN}).")
        return uid


class MemoryConsentRequest(BaseModel):
    """Mock/dev consent callback — replace with application backend in production."""

    user_id: str
    proposal_id: str
    decision: Literal["accepted", "declined"]


class MemoryConsentResponse(BaseModel):
    user_id: str
    proposal_id: str
    applied: bool
    summary_updated: bool
    memory_summary: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    mode: str
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class TranscribeResponse(BaseModel):
    transcription: str


class VoiceResponse(BaseModel):
    """JSON voice turn; audio is base64 WAV for easy client playback."""

    session_id: str
    transcription: str
    response_text: str
    mode: str
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    audio_base64: str = Field(
        description="WAV audio of the assistant response (base64-encoded).",
    )
    audio_media_type: str = Field(default="audio/wav")