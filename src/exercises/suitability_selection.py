"""Bounded LLM selection among backend-eligible exercise candidates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from src.llm.response_parser import InteractionIntent

if TYPE_CHECKING:
    from src.conversation.session import EligibleExerciseCandidate

# Serialized JSON size cap for suitability user payload (characters, not tokens).
MAX_SUITABILITY_PAYLOAD_CHARS = 12_000

SUITABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "recommended_exercise_id": {"type": ["string", "null"]},
    },
    "required": ["recommended_exercise_id"],
}

SUITABILITY_SYSTEM = (
    "You decide whether to optionally OFFER one wellness exercise from a verified candidate list. "
    "This is a proposal only, not authorization to start an exercise. "
    "Return recommended_exercise_id from the supplied candidates when one exercise is genuinely appropriate "
    "for the user's current message and recent conversation context; otherwise return null. "
    "If the user declined or refused exercises recently, return null. "
    "Never invent an id. Never return an id outside the candidate list. "
    "Respond only with JSON matching the schema."
)


def suitability_payload_char_len(payload: Dict[str, Any]) -> int:
    """Length of the JSON-serialized suitability user payload (character count)."""
    return len(json.dumps(payload, ensure_ascii=False))


def build_bounded_suitability_payload(
    *,
    user_text: str,
    interaction_intent: Optional[InteractionIntent],
    language: Optional[str],
    candidates: Sequence["EligibleExerciseCandidate"],
    conversation_context: dict,
    max_chars: int = MAX_SUITABILITY_PAYLOAD_CHARS,
) -> Optional[Dict[str, Any]]:
    """
    Build suitability input JSON content within max_chars.
    Preserves full user_message and every candidate id/category; trims context first,
    then rolling summary, then purpose_summary text. Returns None if impossible.
    """
    user_message = user_text.strip()
    intent_val = interaction_intent.value if interaction_intent else None

    minimal_exercises = [
        {"id": c.id, "category": c.category.value} for c in candidates
    ]
    minimal: Dict[str, Any] = {
        "conversation_language": language,
        "user_message": user_message,
        "interaction_intent": intent_val,
        "rolling_summary": None,
        "recent_turns": [],
        "eligible_exercises": minimal_exercises,
    }
    if suitability_payload_char_len(minimal) > max_chars:
        return None

    recent_turns: List[Dict[str, Any]] = [
        {"role": t.get("role"), "content": (t.get("content") or "").strip()}
        for t in (conversation_context.get("recent_turns") or [])
        if (t.get("content") or "").strip()
    ]
    rolling = conversation_context.get("rolling_summary")
    rolling_summary = (rolling.strip() if isinstance(rolling, str) and rolling.strip() else None)
    purpose_by_id: Dict[str, str] = {
        c.id: (c.purpose_summary or "").strip() for c in candidates
    }

    def assemble() -> Dict[str, Any]:
        exercises: List[Dict[str, Any]] = []
        for c in candidates:
            item: Dict[str, Any] = {"id": c.id, "category": c.category.value}
            ps = purpose_by_id.get(c.id, "")
            if ps:
                item["purpose_summary"] = ps
            exercises.append(item)
        return {
            "conversation_language": language,
            "user_message": user_message,
            "interaction_intent": intent_val,
            "rolling_summary": rolling_summary,
            "recent_turns": recent_turns,
            "eligible_exercises": exercises,
        }

    payload = assemble()
    if suitability_payload_char_len(payload) <= max_chars:
        return payload

    # Trim older conversation turns first (list is chronological; drop from front).
    while len(recent_turns) > 0:
        recent_turns.pop(0)
        payload = assemble()
        if suitability_payload_char_len(payload) <= max_chars:
            return payload

    # Trim rolling summary from the end.
    summary_text = rolling_summary or ""
    while summary_text:
        cut = min(400, max(1, len(summary_text) // 4))
        summary_text = summary_text[: len(summary_text) - cut]
        rolling_summary = summary_text.strip() or None
        payload = assemble()
        if suitability_payload_char_len(payload) <= max_chars:
            return payload

    rolling_summary = None

    # Shorten purpose summaries (never drop candidate ids/categories).
    guard = 0
    while guard < 50_000:
        guard += 1
        payload = assemble()
        if suitability_payload_char_len(payload) <= max_chars:
            return payload
        active = [c.id for c in candidates if purpose_by_id.get(c.id)]
        if not active:
            return None
        longest_id = max(active, key=lambda cid: len(purpose_by_id[cid]))
        text = purpose_by_id[longest_id]
        cut = min(200, max(1, len(text) // 4))
        purpose_by_id[longest_id] = text[: len(text) - cut].strip()

    return None


def select_exercise_candidate(
    llm,
    *,
    user_text: str,
    interaction_intent: Optional[InteractionIntent],
    language: Optional[str],
    candidates: Sequence["EligibleExerciseCandidate"],
    conversation_context: dict,
) -> Optional[str]:
    if llm is None or not hasattr(llm, "generate_auxiliary_json"):
        return None
    if not candidates:
        return None

    allowed_ids = {c.id for c in candidates}
    payload = build_bounded_suitability_payload(
        user_text=user_text,
        interaction_intent=interaction_intent,
        language=language,
        candidates=candidates,
        conversation_context=conversation_context,
    )
    if payload is None:
        return None

    messages = [
        {"role": "system", "content": SUITABILITY_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        data = llm.generate_auxiliary_json(
            messages, schema=SUITABILITY_SCHEMA, max_tokens=96
        )
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    selected = data.get("recommended_exercise_id")
    if selected is None:
        return None
    if not isinstance(selected, str):
        return None
    selected = selected.strip()
    if not selected or selected not in allowed_ids:
        return None
    return selected