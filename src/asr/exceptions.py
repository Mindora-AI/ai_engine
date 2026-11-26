"""ASR routing and dependency errors."""


class UnsupportedASRLanguageError(ValueError):
    """Raised when the requested ASR language has no configured backend."""


class WhisperASRNotAvailableError(ImportError):
    """Raised when English/French ASR is requested but faster-whisper is not installed."""