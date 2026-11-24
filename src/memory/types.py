"""Structured types for memory proposals and approved summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MemoryCategory(str, Enum):
    CIRCUMSTANCE = "circumstance"
    PREFERENCE = "preference"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MemoryProposal:
    """Candidate persistent memory — not stored until explicitly approved."""

    proposal_id: str
    content: str
    rationale: str
    category: MemoryCategory = MemoryCategory.OTHER
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    source_language: Optional[str] = None
    source_turn_snippet: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "content": self.content,
            "rationale": self.rationale,
            "category": self.category.value,
            "confidence": self.confidence.value,
            "source_language": self.source_language,
        }

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())


@dataclass
class PersistentMemorySummary:
    """Compact approved memory for a user (not a full transcript)."""

    user_id: str
    summary_text: str
    updated_at: Optional[str] = None
    version: int = 1


@dataclass
class PendingContextFold:
    """Rolling summary fold applied only after a successful assistant turn."""

    removed_history: List[Dict[str, str]] = field(default_factory=list)
    new_rolling_summary: str = ""
    retained_history: List[Dict[str, str]] = field(default_factory=list)