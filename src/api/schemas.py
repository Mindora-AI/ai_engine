"""API request/response models (public contract only)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    engine_initialized: bool


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

    @field_validator("user_text")
    @classmethod
    def user_text_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_text must not be empty")
        return stripped


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