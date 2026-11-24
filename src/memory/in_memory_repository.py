"""Development/test memory store — not for production."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, Optional

from src.memory.types import PersistentMemorySummary


class InMemoryMemoryRepository:
    """Thread-safe in-memory memory summaries isolated by user_id."""

    def __init__(self) -> None:
        self._data: Dict[str, PersistentMemorySummary] = {}
        self._lock = threading.Lock()

    def get_summary(self, user_id: str) -> Optional[PersistentMemorySummary]:
        with self._lock:
            return self._data.get(user_id)

    def save_summary(self, record: PersistentMemorySummary) -> None:
        with self._lock:
            self._data[record.user_id] = record

    def update_summary(self, user_id: str, summary_text: str) -> PersistentMemorySummary:
        with self._lock:
            existing = self._data.get(user_id)
            version = (existing.version + 1) if existing else 1
            record = PersistentMemorySummary(
                user_id=user_id,
                summary_text=summary_text.strip(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                version=version,
            )
            self._data[user_id] = record
            return record

    def delete_memory(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._data:
                del self._data[user_id]
                return True
            return False