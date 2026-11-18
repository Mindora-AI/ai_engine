"""
MWIZERWA AI — Conversation Domain Layer
Defines lightweight domain models and enums for conversational turns and sessions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from src.utils.language_detector import LanguageDetectionResult


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


@dataclass
class TurnRequest:
    """
    Represents an incoming conversational turn.
    """
    user_text: str
    mode: Optional[ConversationMode] = None
    language: Optional[str] = None
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
    metadata: Dict[str, Any] = field(default_factory=dict)
