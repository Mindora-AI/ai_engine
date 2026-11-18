"""
MWIZERWA AI — Exercise Domain Package
Exposes domain categories, models, session state, and the static exercise catalogue.
"""

from src.exercises.categories import (
    ExerciseCategory,
    ExerciseStatus,
    ExerciseTriggerType,
)
from src.exercises.models import (
    Exercise,
    ExerciseStep,
    ExerciseSessionState,
    UnsupportedLanguageError,
    ExerciseValidationError,
    SUPPORTED_EXERCISE_LANGUAGES,
)
from src.exercises.registry import (
    ExerciseNotFoundError,
    get_exercise,
    get_all_exercises,
    get_exercises_by_category,
    get_available_exercise_ids,
)
from src.exercises.recommender import (
    ExerciseRecommender,
    RecommendationAction,
    ExplicitRequestResult,
    RecommendationDecision,
    CATEGORY_DEFAULT_EXERCISE,
)
from src.exercises.lifecycle import (
    ExerciseLifecycle,
    LifecycleResult,
    UserResponseType,
    STOP_ACKNOWLEDGMENTS,
    DECLINE_ACKNOWLEDGMENTS,
    UNRELATED_CHECKINS,
    OFFER_TEMPLATES,
)

__all__ = [
    "ExerciseCategory",
    "ExerciseStatus",
    "ExerciseTriggerType",
    "Exercise",
    "ExerciseStep",
    "ExerciseSessionState",
    "UnsupportedLanguageError",
    "ExerciseValidationError",
    "SUPPORTED_EXERCISE_LANGUAGES",
    "ExerciseNotFoundError",
    "get_exercise",
    "get_all_exercises",
    "get_exercises_by_category",
    "get_available_exercise_ids",
    "ExerciseRecommender",
    "RecommendationAction",
    "ExplicitRequestResult",
    "RecommendationDecision",
    "CATEGORY_DEFAULT_EXERCISE",
    "ExerciseLifecycle",
    "LifecycleResult",
    "UserResponseType",
    "STOP_ACKNOWLEDGMENTS",
    "DECLINE_ACKNOWLEDGMENTS",
    "UNRELATED_CHECKINS",
    "OFFER_TEMPLATES",
]
