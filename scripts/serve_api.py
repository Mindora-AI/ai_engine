#!/usr/bin/env python3
"""Run MWIZERWA FastAPI server (loads GGUF once at startup)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import uvicorn

    host = os.environ.get("MWIZERWA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("MWIZERWA_API_PORT", "8000"))
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()