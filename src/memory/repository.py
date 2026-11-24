"""Replaceable persistence interface (mock or application database adapter)."""

from __future__ import annotations

from typing import Optional, Protocol

from src.memory.types import PersistentMemorySummary


class MemoryRepository(Protocol):
    """Application replaces this with a trusted backend store."""

    def get_summary(self, user_id: str) -> Optional[PersistentMemorySummary]:
        ...

    def save_summary(self, record: PersistentMemorySummary) -> None:
        ...

    def update_summary(self, user_id: str, summary_text: str) -> PersistentMemorySummary:
        ...

    def delete_memory(self, user_id: str) -> bool:
        ...