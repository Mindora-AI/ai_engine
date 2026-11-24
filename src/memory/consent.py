"""Consent boundary — AI proposes; application (or mock) records user decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Protocol, Set, Tuple
import threading

from src.memory.types import MemoryProposal


class ConsentDecision(str, Enum):
    ACCEPTED = "accepted"
    DECLINED = "declined"


@dataclass
class ConsentRecord:
    proposal_id: str
    user_id: str
    decision: ConsentDecision


class ConsentGateway(Protocol):
    """Production: implemented by Mwizerwa app backend after user UI interaction."""

    def register_pending(self, user_id: str, proposal: MemoryProposal) -> None:
        ...

    def fetch_pending(self, user_id: str, proposal_id: str) -> Optional[MemoryProposal]:
        ...

    def finalize(self, user_id: str, proposal_id: str, decision: ConsentDecision) -> bool:
        ...

    def is_resolved(self, proposal_id: str) -> bool:
        ...


class MockConsentGateway:
    """
    Test/dev consent simulator. Never auto-accepts proposals.
    Tracks pending and resolved proposal ids to avoid duplicate permission prompts.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, Tuple[str, MemoryProposal]] = {}
        self._resolved: Dict[str, ConsentRecord] = {}
        self._offered_fingerprints: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _fingerprint(proposal: MemoryProposal) -> str:
        return f"{proposal.category.value}:{proposal.content.strip().lower()}"

    def register_pending(self, user_id: str, proposal: MemoryProposal) -> None:
        with self._lock:
            self._pending[proposal.proposal_id] = (user_id, proposal)
            self._offered_fingerprints.setdefault(user_id, set()).add(
                self._fingerprint(proposal)
            )

    def should_offer(self, user_id: str, proposal: MemoryProposal) -> bool:
        """Skip if the same content was already offered for consent."""
        with self._lock:
            return self._fingerprint(proposal) not in self._offered_fingerprints.get(
                user_id, set()
            )

    def fetch_pending(self, user_id: str, proposal_id: str) -> Optional[MemoryProposal]:
        with self._lock:
            if proposal_id in self._resolved:
                return None
            entry = self._pending.get(proposal_id)
            if entry is None:
                return None
            owner_id, proposal = entry
            if owner_id != user_id:
                return None
            return proposal

    def finalize(self, user_id: str, proposal_id: str, decision: ConsentDecision) -> bool:
        with self._lock:
            if proposal_id in self._resolved:
                return False
            entry = self._pending.get(proposal_id)
            if entry is None:
                return False
            owner_id, _proposal = entry
            if owner_id != user_id:
                return False
            del self._pending[proposal_id]
            self._resolved[proposal_id] = ConsentRecord(
                proposal_id=proposal_id,
                user_id=user_id,
                decision=decision,
            )
            return True

    def invalidate_user_pending(self, user_id: str) -> int:
        """Remove outstanding proposals for a user (e.g. after memory deletion)."""
        with self._lock:
            to_drop = [
                pid for pid, (uid, _) in self._pending.items() if uid == user_id
            ]
            for pid in to_drop:
                del self._pending[pid]
            return len(to_drop)

    def is_resolved(self, proposal_id: str) -> bool:
        with self._lock:
            return proposal_id in self._resolved

    def get_decision(self, proposal_id: str) -> Optional[ConsentRecord]:
        with self._lock:
            return self._resolved.get(proposal_id)