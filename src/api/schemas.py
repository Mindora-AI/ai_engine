"""API request/response models (public contract only)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.exercises.categories import ExerciseCategory


class HealthResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    engine_initialized: bool
    backends: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cheap readiness: dependency, artifact hints, and in-memory loaded flags.",
    )


SUPPORTED_CHAT_LANGUAGES = frozenset({"rw", "en", "fr"})
_MAX_SESSION_ID_LEN = 128
_MAX_USER_ID_LEN = 128

ExerciseCategoryLiteral = Literal[
    "breathing",
    "grounding",
    "thought_reflection",
    "self_compassion",
    "gratitude_journaling",
]


class RecentTurnMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class EligibleExerciseDescriptor(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    category: ExerciseCategoryLiteral
    purpose_summary: Optional[str] = Field(
        default=None,
        description="Short backend-owned description of what this exercise helps with.",
        max_length=500,
    )

    @field_validator("category")
    @classmethod
    def category_valid(cls, value: str) -> str:
        try:
            ExerciseCategory(value)
        except ValueError as e:
            supported = ", ".join(c.value for c in ExerciseCategory)
            raise ValueError(
                f"Unsupported exercise category '{value}'. Supported: {supported}."
            ) from e
        return value


class TurnContextRequest(BaseModel):
    """
    Optional trusted backend context for a single chat turn.
    Omitted fields are not interpreted as overrides.
    """

    approved_memory_summary: Optional[str] = Field(
        default=None,
        description=(
            "Authoritative approved memory text from the backend. "
            "Send an empty string when the user has no approved memory; omit the field "
            "to leave memory resolution to standalone/dev fallbacks."
        ),
    )
    recent_turns: Optional[List[RecentTurnMessage]] = Field(
        default=None,
        description="Bounded backend-owned conversation history for inference alignment.",
    )
    eligible_exercises: Optional[List[EligibleExerciseDescriptor]] = Field(
        default=None,
        description=(
            "Backend-filtered exercise candidates (id + category). "
            "An empty list means no exercise may be recommended this turn."
        ),
    )


class SafetySignal(BaseModel):
    """Structured safety metadata when Gate 1 and/or Gate 2 fired."""

    gate1_triggered: bool
    gate2_triggered: bool
    safety_action: str
    category: Optional[str] = None
    subject: Optional[str] = None
    temporal: Optional[str] = None
    intent: Optional[str] = None
    immediacy: Optional[str] = None
    detected_expression_language: Optional[str] = None
    gate2_violation_category: Optional[str] = None


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
    context: Optional[TurnContextRequest] = Field(
        default=None,
        description="Optional backend-supplied application context for this turn.",
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
    recommended_exercise_id: Optional[str] = Field(
        default=None,
        description="Stable backend exercise id when an unsolicited exercise was offered.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class TranscribeResponse(BaseModel):
    transcription: str
    language: Optional[str] = Field(
        default=None,
        description="ASR language used (rw/en/fr). Omitted language resolves to rw.",
    )


class VoiceResponse(BaseModel):
    """JSON voice turn; audio is base64 WAV for easy client playback."""

    session_id: str
    transcription: str
    response_text: str
    mode: str
    language: Optional[str] = None
    asr_language: Optional[str] = Field(
        default=None,
        description="Language code used for speech recognition (rw/en/fr).",
    )
    recommended_exercise_id: Optional[str] = Field(
        default=None,
        description="Stable backend exercise id when an unsolicited exercise was offered.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    audio_base64: Optional[str] = Field(
        default=None,
        description="WAV audio of the assistant response (base64-encoded); null when TTS failed.",
    )
    audio_status: Literal["success", "failed"] = Field(
        default="success",
        description="Whether assistant speech was synthesized successfully.",
    )
    audio_error: Optional[str] = Field(
        default=None,
        description="Safe client message when audio_status is failed.",
    )
    audio_media_type: Optional[str] = Field(default="audio/wav")