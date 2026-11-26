"""Map HTTP context models to conversation TurnContext."""

from __future__ import annotations

from typing import List, Optional

from src.api.schemas import TurnContextRequest
from src.conversation.context_integration import (
    TurnContextValidationError,
    validate_approved_memory_summary,
    validate_eligible_exercises,
    validate_recent_turns,
)
from src.conversation.session import TurnContext


def build_turn_context(request: Optional[TurnContextRequest]) -> Optional[TurnContext]:
    if request is None:
        return None

    fields_set = request.model_fields_set
    approved: Optional[str] = None
    backend_memory = False
    if "approved_memory_summary" in fields_set:
        backend_memory = True
        approved = validate_approved_memory_summary(request.approved_memory_summary or "")

    recent: Optional[List[dict]] = None
    if "recent_turns" in fields_set and request.recent_turns is not None:
        recent = validate_recent_turns(
            [{"role": m.role, "content": m.content} for m in request.recent_turns]
        )

    eligible = None
    if "eligible_exercises" in fields_set:
        raw = request.eligible_exercises or []
        eligible = validate_eligible_exercises(
            [
                {
                    "id": e.id,
                    "category": e.category,
                    "purpose_summary": e.purpose_summary,
                }
                for e in raw
            ]
        )

    return TurnContext(
        approved_memory_summary=approved,
        recent_turns=recent,
        eligible_exercises=eligible,
        backend_supplied_approved_memory=backend_memory,
    )


def build_turn_context_from_json(data: dict) -> TurnContext:
    """Validate arbitrary JSON (e.g. voice multipart context) into TurnContext."""
    try:
        parsed = TurnContextRequest.model_validate(data)
    except Exception as e:
        raise TurnContextValidationError(str(e)) from e
    ctx = build_turn_context(parsed)
    if ctx is None:
        raise TurnContextValidationError("context must include at least one field.")
    return ctx