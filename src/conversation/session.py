"""
MWIZERWA AI — Conversation Domain Layer
Defines lightweight domain models and enums for conversational turns and sessions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from src.utils.language_detector import LanguageDetectionResult
from src.exercises.categories import ExerciseCategory
from src.exercises.models import ExerciseSessionState


class ConversationMode(str, Enum):
    """
    Domain vocabulary for conversational delivery modes.
    Mode-specific prompting and behavioral adaptations are implemented in subsequent chunks.
    """
    GENTLE = "gentle"
    NORMAL = "normal"
    MOTIVATIONAL = "motivational"


@dataclass
class SessionState:
    """
    Lightweight representation of session-level conversational state.
    Important: Message history is managed exclusively by ConversationManager
    and is not duplicated here.
    """
    mode: ConversationMode = ConversationMode.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    exercise_state: ExerciseSessionState = field(default_factory=ExerciseSessionState)


@dataclass(frozen=True)
class EligibleExerciseCandidate:
    """
    Backend-supplied descriptor of an eligible, verified exercise candidate.
    """
    id: str
    category: ExerciseCategory
    purpose_summary: Optional[str] = None


@dataclass
class TurnContext:
    """
    Optional backend-supplied trusted application context for a turn.
    """
    approved_memory_summary: Optional[str] = None
    recent_turns: Optional[List[Dict[str, str]]] = None
    eligible_exercises: Optional[List[EligibleExerciseCandidate]] = None
    backend_supplied_approved_memory: bool = False


@dataclass
class TurnRequest:
    """
    Represents an incoming conversational turn.
    """
    user_text: str
    mode: Optional[ConversationMode] = None
    language: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[TurnContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    """
    Represents the result of processing a conversational turn in a predictable, typed structure.
    """
    response_text: str
    mode: ConversationMode
    language: Optional[str] = None
    language_result: Optional[LanguageDetectionResult] = None
    recommended_exercise_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
