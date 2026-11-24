"""Orchestrates proposals, consent, and approved persistent memory."""

from __future__ import annotations

from typing import List, Optional

from src.memory.consent import ConsentDecision, ConsentGateway, MockConsentGateway
from src.memory.extraction import extract_memory_proposals
from src.memory.in_memory_repository import InMemoryMemoryRepository
from src.memory.repository import MemoryRepository
from src.memory.summary_merge import merge_approved_into_summary
from src.memory.types import MemoryProposal


class MemoryCoordinator:
    """
    AI-side memory workflow. The application must supply trusted user_id values
    and implement ConsentGateway for production consent UX.
    """

    def __init__(
        self,
        repository: Optional[MemoryRepository] = None,
        consent: Optional[ConsentGateway] = None,
        llm=None,
    ) -> None:
        self._repo = repository or InMemoryMemoryRepository()
        self._consent = consent or MockConsentGateway()
        self._llm = llm

    def set_llm(self, llm) -> None:
        self._llm = llm

    def get_approved_summary_text(self, user_id: Optional[str]) -> str:
        if not user_id:
            return ""
        record = self._repo.get_summary(user_id)
        return record.summary_text.strip() if record else ""

    def propose_from_turn(
        self,
        user_id: Optional[str],
        user_text: str,
        assistant_text: str,
        language: Optional[str] = None,
    ) -> List[MemoryProposal]:
        if not user_id or self._llm is None:
            return []

        proposals = extract_memory_proposals(
            self._llm,
            user_text=user_text,
            assistant_text=assistant_text,
            language=language,
        )
        offered: List[MemoryProposal] = []
        gateway = self._consent
        for p in proposals:
            if hasattr(gateway, "should_offer") and not gateway.should_offer(user_id, p):
                continue
            gateway.register_pending(user_id, p)
            offered.append(p)
        return offered

    def apply_consent_decision(
        self,
        user_id: str,
        proposal_id: str,
        decision: ConsentDecision,
        proposal: Optional[MemoryProposal] = None,
    ) -> Optional[str]:
        """
        Returns updated summary text when accepted; None when declined or invalid.
        Never persists without an explicit accepted decision.
        """
        if self._consent.is_resolved(proposal_id):
            return None

        pending = proposal
        if pending is None and hasattr(self._consent, "fetch_pending"):
            pending = self._consent.fetch_pending(user_id, proposal_id)
        if pending is None:
            return None

        if decision == ConsentDecision.DECLINED:
            if hasattr(self._consent, "finalize"):
                self._consent.finalize(user_id, proposal_id, decision)
            return None

        if decision != ConsentDecision.ACCEPTED:
            return None

        content = pending.content.strip()
        if not content:
            return None

        existing = self.get_approved_summary_text(user_id)
        merged = merge_approved_into_summary(
            self._llm,
            existing,
            [content],
        )
        if merged is None:
            return None

        self._repo.update_summary(user_id, merged)
        if hasattr(self._consent, "finalize"):
            self._consent.finalize(user_id, proposal_id, ConsentDecision.ACCEPTED)
        return merged

    def delete_user_memory(self, user_id: str) -> bool:
        deleted = self._repo.delete_memory(user_id)
        if hasattr(self._consent, "invalidate_user_pending"):
            self._consent.invalidate_user_pending(user_id)
        return deleted