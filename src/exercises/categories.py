"""
MWIZERWA AI — Exercise Domain Categories and Enums
Defines domain vocabulary, statuses, and triggers for self-guided wellness exercises.
"""

from enum import Enum


class ExerciseCategory(str, Enum):
    """
    Core non-clinical wellness exercise categories.
    """
    BREATHING = "breathing"
    GROUNDING = "grounding"
    THOUGHT_REFLECTION = "thought_reflection"
    SELF_COMPASSION = "self_compassion"
    GRATITUDE_JOURNALING = "gratitude_journaling"


class ExerciseStatus(str, Enum):
    """
    Lifecycle states for an exercise session.
    """
    IDLE = "idle"
    OFFERED = "offered"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DECLINED = "declined"
    ABANDONED = "abandoned"


class ExerciseTriggerType(str, Enum):
    """
    Mechanism by which an exercise was initiated or presented.
    """
    EXPLICIT_USER_REQUEST = "explicit_user_request"
    UNSOLICITED_OFFER = "unsolicited_offer"
