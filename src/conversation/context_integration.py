"""
Validation and application of backend-supplied turn context.

Recent-turn contract (backend → AI):
- `recent_turns` must list completed user/assistant pairs from the backend transcript.
- It must NOT include the current request's `user_text` (the engine appends that during inference).
- The latest assistant reply should be included when the backend has already persisted it.
- Identical consecutive payloads are ignored (fingerprint deduplication).
- When the fingerprint changes, verbatim history is replaced and AI-owned rolling summaries
  are cleared (see ConversationEngine._apply_backend_recent_turns).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence

from src.conversation.session import EligibleExerciseCandidate, TurnContext, TurnRequest
from src.exercises.categories import ExerciseCategory

ALLOWED_TURN_ROLES = frozenset({"user", "assistant"})
MAX_RECENT_TURNS = 40
MAX_TURN_CONTENT_CHARS = 12_000
MAX_APPROVED_MEMORY_CHARS = 16_000
MAX_ELIGIBLE_EXERCISES = 32
MAX_EXERCISE_ID_LEN = 128
MAX_PURPOSE_SUMMARY_CHARS = 500


class TurnContextValidationError(ValueError):
    """Raised when backend-supplied context fails validation."""


def validate_recent_turns(turns: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(turns) > MAX_RECENT_TURNS:
        raise TurnContextValidationError(
            f"recent_turns exceeds maximum of {MAX_RECENT_TURNS} messages."
        )
    out: List[Dict[str, str]] = []
    for i, msg in enumerate(turns):
        if not isinstance(msg, dict):
            raise TurnContextValidationError(
                f"recent_turns[{i}] must be an object with role and content."
            )
        role = msg.get("role")
        content = msg.get("content")
        if role not in ALLOWED_TURN_ROLES:
            raise TurnContextValidationError(
                f"recent_turns[{i}].role must be one of: user, assistant."
            )
        if not isinstance(content, str):
            raise TurnContextValidationError(
                f"recent_turns[{i}].content must be a string."
            )
        stripped = content.strip()
        if not stripped:
            raise TurnContextValidationError(
                f"recent_turns[{i}].content must not be empty."
            )
        if len(stripped) > MAX_TURN_CONTENT_CHARS:
            raise TurnContextValidationError(
                f"recent_turns[{i}].content exceeds {MAX_TURN_CONTENT_CHARS} characters."
            )
        out.append({"role": role, "content": stripped})
    return out


def recent_turns_fingerprint(turns: Sequence[Dict[str, str]]) -> str:
    payload = json.dumps(turns, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_approved_memory_summary(
    request: TurnRequest,
    internal_summary_provider,
) -> str:
    """
    Backend-supplied approved memory wins when explicitly provided.
    internal_summary_provider: callable returning internal repo summary for user_id.
    """
    ctx = request.context
    if ctx is not None and ctx.backend_supplied_approved_memory:
        return (ctx.approved_memory_summary or "").strip()
    if request.user_id and internal_summary_provider is not None:
        return (internal_summary_provider() or "").strip()
    return ""


def validate_eligible_exercises(
    raw: Sequence[Dict[str, str]],
) -> List[EligibleExerciseCandidate]:
    if len(raw) > MAX_ELIGIBLE_EXERCISES:
        raise TurnContextValidationError(
            f"eligible_exercises exceeds maximum of {MAX_ELIGIBLE_EXERCISES} items."
        )
    out: List[EligibleExerciseCandidate] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TurnContextValidationError(
                f"eligible_exercises[{i}] must be an object with id and category."
            )
        ex_id = item.get("id")
        category = item.get("category")
        if not isinstance(ex_id, str) or not ex_id.strip():
            raise TurnContextValidationError(
                f"eligible_exercises[{i}].id must be a non-empty string."
            )
        ex_id = ex_id.strip()
        if len(ex_id) > MAX_EXERCISE_ID_LEN:
            raise TurnContextValidationError(
                f"eligible_exercises[{i}].id exceeds {MAX_EXERCISE_ID_LEN} characters."
            )
        if ex_id in seen_ids:
            raise TurnContextValidationError(
                f"eligible_exercises contains duplicate id '{ex_id}'."
            )
        seen_ids.add(ex_id)
        if not isinstance(category, str) or not category.strip():
            raise TurnContextValidationError(
                f"eligible_exercises[{i}].category must be a non-empty string."
            )
        try:
            cat = ExerciseCategory(category.strip().lower())
        except ValueError as e:
            supported = ", ".join(c.value for c in ExerciseCategory)
            raise TurnContextValidationError(
                f"eligible_exercises[{i}].category '{category}' is invalid. "
                f"Supported: {supported}."
            ) from e
        purpose_raw = item.get("purpose_summary")
        purpose: Optional[str] = None
        if purpose_raw is not None:
            if not isinstance(purpose_raw, str):
                raise TurnContextValidationError(
                    f"eligible_exercises[{i}].purpose_summary must be a string when provided."
                )
            purpose = purpose_raw.strip()
            if len(purpose) > MAX_PURPOSE_SUMMARY_CHARS:
                raise TurnContextValidationError(
                    f"eligible_exercises[{i}].purpose_summary exceeds "
                    f"{MAX_PURPOSE_SUMMARY_CHARS} characters."
                )
            if not purpose:
                purpose = None
        out.append(EligibleExerciseCandidate(id=ex_id, category=cat, purpose_summary=purpose))
    return out


def validate_approved_memory_summary(value: str) -> str:
    if len(value) > MAX_APPROVED_MEMORY_CHARS:
        raise TurnContextValidationError(
            f"approved_memory_summary exceeds {MAX_APPROVED_MEMORY_CHARS} characters."
        )
    return value.strip()


def exercise_id_for_category(
    category: ExerciseCategory,
    eligible: Optional[List[EligibleExerciseCandidate]],
    *,
    registry_default: Dict[ExerciseCategory, str],
) -> Optional[str]:
    """Resolve exercise id from backend candidates or dev registry fallback."""
    if eligible is not None:
        if not eligible:
            return None
        for candidate in eligible:
            if candidate.category == category:
                return candidate.id
        return None
    return registry_default.get(category)


def backend_controls_exercise_candidates(ctx: Optional[TurnContext]) -> bool:
    return ctx is not None and ctx.eligible_exercises is not None