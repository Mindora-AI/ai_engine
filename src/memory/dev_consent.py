"""Environment gate for the development-only HTTP memory consent endpoint."""

from __future__ import annotations

import os


def dev_memory_consent_enabled() -> bool:
    """
    Production must leave this disabled. Enable only for local/dev automated tests:
    MWIZERWA_ALLOW_DEV_MEMORY_CONSENT=1 or MWIZERWA_ENV=development|dev|test
    """
    flag = os.environ.get("MWIZERWA_ALLOW_DEV_MEMORY_CONSENT", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    env = os.environ.get("MWIZERWA_ENV", "").strip().lower()
    return env in ("development", "dev", "test", "local")