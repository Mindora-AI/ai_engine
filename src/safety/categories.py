"""
MWIZERWA AI — Safety Domain Models
Defines immutable domain structures, enums, and safety assessment contracts for Chunk 4.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class SafetyCategory(str, Enum):
    """
    Core safety classification categories distinguishing ordinary conversation
    and emotional distress from acute crisis, self-harm, and violent situations.
    """
    # Non-Crisis / Standard Wellness Dialogue
    SAFE_CONVERSATIONAL = "safe_conversational"
    ORDINARY_DISTRESS = "ordinary_distress"

    # Ambiguity / Exploration
    AMBIGUOUS_DISTRESS = "ambiguous_distress"
    GENERAL_OR_HYPOTHETICAL = "general_or_hypothetical"

    # Self-Harm & Suicide Spectrum
    PASSIVE_DEATH_WISH = "passive_death_wish"
    SELF_HARM_THOUGHTS = "self_harm_thoughts"
    IMMINENT_SELF_HARM = "imminent_self_harm"

    # Violence & Harm to Others
    ANGER_OR_VENTING = "anger_or_venting"
    THREAT_OR_INTENT_HARM_OTHERS = "threat_or_intent_harm_others"
    IMMINENT_VIOLENCE_OTHERS = "imminent_violence_others"

    # Physical Danger / Medical Traumas
    THIRD_PERSON_CRISIS = "third_person_crisis"
    ONGOING_PHYSICAL_DANGER = "ongoing_physical_danger"
    ACUTE_MEDICAL_EMERGENCY = "acute_medical_emergency"


class ApplicationAction(str, Enum):
    """
    Decoupled application-level actions translating safety assessments into
    concrete engine execution behaviors.
    """
    NORMAL_CONVERSATION = "normal_conversation"
    SUPPORTIVE_CLARIFICATION = "supportive_clarification"
    SAFETY_FOCUSED_RESPONSE = "safety_focused_response"
    DETERMINISTIC_EMERGENCY_OVERRIDE = "deterministic_emergency_override"


class SubjectTarget(str, Enum):
    """Identifies the subject or target of the disclosed risk."""
    SELF = "self"
    THIRD_PERSON_KNOWN = "third_person_known"
    THIRD_PERSON_GENERAL = "third_person_general"
    UNSPECIFIED = "unspecified"


class TemporalStatus(str, Enum):
    """Identifies the time horizon of the disclosed experience."""
    HISTORICAL = "historical"
    CURRENT = "current"
    HYPOTHETICAL = "hypothetical"


class IntentLevel(str, Enum):
    """Identifies the explicit degree of intent or planning."""
    NONE = "none"
    PASSIVE_WISH = "passive_wish"
    INTRUSIVE_THOUGHTS = "intrusive_thoughts"
    FEAR_OF_ACTING = "fear_of_acting"
    EXPLICIT_INTENT = "explicit_intent"
    ACTION_IN_PROGRESS = "action_in_progress"


class MeansStatus(str, Enum):
    """Identifies whether means of self-harm or violence are mentioned, accessible, or active."""
    NONE_MENTIONED = "none_mentioned"
    MENTION_ONLY = "mention_only"
    ACCESS_AVAILABLE = "access_available"
    PREPARATION = "preparation"
    INGESTION_OR_INJURY_UNDERWAY = "underway"


class ImmediacyLevel(str, Enum):
    """Identifies the temporal urgency of the situation."""
    NONE = "none"
    UNCLEAR = "unclear"
    IMMINENT = "imminent"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class SafetyAssessment:
    """
    Immutable representation of an independent safety evaluation.
    Captures multi-dimensional risk context without mutable collections.
    """
    category: SafetyCategory
    subject: SubjectTarget = SubjectTarget.UNSPECIFIED
    temporal: TemporalStatus = TemporalStatus.CURRENT
    intent: IntentLevel = IntentLevel.NONE
    means: MeansStatus = MeansStatus.NONE_MENTIONED
    immediacy: ImmediacyLevel = ImmediacyLevel.NONE
    detected_expression_language: Optional[str] = None
    matched_markers: Tuple[str, ...] = ()

    def __post_init__(self):
        valid_languages = ("rw", "en", "fr", None)
        if self.detected_expression_language not in valid_languages:
            raise ValueError(
                f"Invalid detected_expression_language: '{self.detected_expression_language}'. "
                f"Supported values are: {valid_languages}."
            )
        if not isinstance(self.matched_markers, tuple):
            object.__setattr__(self, "matched_markers", tuple(self.matched_markers))
