"""AI-side memory proposal extraction (requires explicit user consent to persist)."""

from __future__ import annotations

import json
from typing import List, Optional

from src.memory.prompts import MEMORY_EXTRACTION_SYSTEM
from src.memory.types import ConfidenceLevel, MemoryCategory, MemoryProposal

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "rationale": {"type": "string"},
                    "category": {"type": "string"},
                    "confidence": {"type": "string"},
                    "source_turn_snippet": {"type": "string"},
                },
                "required": ["content", "rationale"],
            },
        }
    },
    "required": ["proposals"],
}

_CATEGORY_MAP = {
    "circumstance": MemoryCategory.CIRCUMSTANCE,
    "preference": MemoryCategory.PREFERENCE,
    "goal": MemoryCategory.GOAL,
    "relationship": MemoryCategory.RELATIONSHIP,
    "other": MemoryCategory.OTHER,
}

_CONFIDENCE_MAP = {
    "low": ConfidenceLevel.LOW,
    "medium": ConfidenceLevel.MEDIUM,
    "high": ConfidenceLevel.HIGH,
}


def extract_memory_proposals(
    llm,
  *,
    user_text: str,
    assistant_text: str,
    language: Optional[str] = None,
) -> List[MemoryProposal]:
    if not user_text.strip() or not assistant_text.strip():
        return []

    payload = {
        "user_message": user_text.strip(),
        "assistant_message": assistant_text.strip(),
        "conversation_language": language,
    }
    messages = [
        {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    if not hasattr(llm, "generate_auxiliary_json"):
        return []

    try:
        data = llm.generate_auxiliary_json(
            messages,
            schema=EXTRACTION_SCHEMA,
            max_tokens=320,
        )
    except Exception:
        return []

    out: List[MemoryProposal] = []
    for raw in data.get("proposals") or []:
        if not isinstance(raw, dict):
            continue
        content = (raw.get("content") or "").strip()
        rationale = (raw.get("rationale") or "").strip()
        if not content or not rationale:
            continue
        if len(content) < 8:
            continue
        cat = _CATEGORY_MAP.get(
            str(raw.get("category", "other")).strip().lower(),
            MemoryCategory.OTHER,
        )
        conf = _CONFIDENCE_MAP.get(
            str(raw.get("confidence", "low")).strip().lower(),
            ConfidenceLevel.LOW,
        )
        snippet = (raw.get("source_turn_snippet") or user_text)[:240]
        out.append(
            MemoryProposal(
                proposal_id=MemoryProposal.new_id(),
                content=content,
                rationale=rationale,
                category=cat,
                confidence=conf,
                source_language=language,
                source_turn_snippet=snippet,
            )
        )
    return out