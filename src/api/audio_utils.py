"""Upload validation and temporary audio file helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Set

from fastapi import HTTPException, UploadFile, status

def max_upload_bytes() -> int:
    return int(os.environ.get("MWIZERWA_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


DEFAULT_MAX_UPLOAD_BYTES = max_upload_bytes()

ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".webm", ".mp3", ".ogg", ".flac", ".m4a", ".mpeg", ".mp4"}
ALLOWED_AUDIO_CONTENT_PREFIXES = ("audio/", "application/octet-stream")


def validate_upload(upload: UploadFile, max_bytes: int | None = None) -> None:
    if max_bytes is None:
        max_bytes = max_upload_bytes()
    if upload.filename:
        ext = Path(upload.filename).suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
            )
    content_type = (upload.content_type or "").lower()
    if content_type and not any(
        content_type.startswith(p) for p in ALLOWED_AUDIO_CONTENT_PREFIXES
    ):
        if not content_type.startswith("video/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported content type '{content_type}'.",
            )


def write_upload_to_tempfile(
    upload: UploadFile,
    *,
    max_bytes: int | None = None,
) -> Path:
    """Stream upload to a temp file; enforce size limit."""
    if max_bytes is None:
        max_bytes = max_upload_bytes()
    validate_upload(upload, max_bytes=max_bytes)
    suffix = Path(upload.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        suffix = ".wav"
    fd, raw_path = tempfile.mkstemp(prefix="mwizerwa_upload_", suffix=suffix)
    path = Path(raw_path)
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Audio upload exceeds limit of {max_bytes // (1024 * 1024)} MB.",
                    )
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def temp_output_wav() -> Path:
    fd, raw = tempfile.mkstemp(prefix="mwizerwa_tts_", suffix=".wav")
    os.close(fd)
    return Path(raw)