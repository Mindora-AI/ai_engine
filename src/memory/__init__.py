"""Persistent memory (AI-side proposals, summaries, mock storage)."""

from src.memory.types import MemoryProposal, PersistentMemorySummary
from src.memory.repository import MemoryRepository
from src.memory.in_memory_repository import InMemoryMemoryRepository
from src.memory.consent import ConsentDecision, ConsentGateway, MockConsentGateway
from src.memory.coordinator import MemoryCoordinator

__all__ = [
    "MemoryProposal",
    "PersistentMemorySummary",
    "MemoryRepository",
    "InMemoryMemoryRepository",
    "ConsentDecision",
    "ConsentGateway",
    "MockConsentGateway",
    "MemoryCoordinator",
]