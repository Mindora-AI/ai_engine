"""Singleton service: one shared LLM, per-session ConversationEngine instances."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.asr.asr_engine import ASREngine, resolve_asr_language
from src.asr.exceptions import UnsupportedASRLanguageError, WhisperASRNotAvailableError
from src.conversation.engine import ConversationEngine
from src.conversation.session import ConversationMode, TurnContext, TurnRequest, TurnResult
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.conversation import ConversationManager
from src.tts.tts_engine import TTSEngine, TTSSynthesisError, UnsupportedTTSLanguageError

from src.api.audio_utils import safe_unlink, temp_output_wav
from src.memory.coordinator import MemoryCoordinator

logger = logging.getLogger(__name__)

# Metadata keys safe to expose via HTTP (no objects, paths, or raw assessments).
PUBLIC_METADATA_KEYS = frozenset(
    {
        "interaction_intent",
        "requires_clarification",
        "detected_language",
        "effective_language",
        "language_confidence",
        "is_language_reliable",
        "is_language_mixed",
        "safety_action",
        "gate1_triggered",
        "gate2_triggered",
        "exercise_status",
        "active_exercise_id",
        "current_step_index",
        "cooldown_turns_remaining",
        "memory_proposals",
        "rolling_summary_active",
        "safety_signal",
    }
)


def _public_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in PUBLIC_METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None:
            continue
        out[key] = _normalize_metadata_public_value(value)
    return out


def _normalize_metadata_public_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {k: _normalize_metadata_public_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_metadata_public_value(v) for v in value]
    return value


class MWIZERWAApiService:
    def __init__(self) -> None:
        self._llm: Optional[AfriqueQwenLLM] = None
        self._tts: Optional[TTSEngine] = None
        self._asr: Optional[ASREngine] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._sessions: Dict[str, ConversationEngine] = {}
        self._sessions_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        self._tts_lock = threading.RLock()
        self._asr_lock = threading.RLock()
        self._memory = MemoryCoordinator()

    @property
    def memory_coordinator(self) -> MemoryCoordinator:
        return self._memory

    @property
    def engine_initialized(self) -> bool:
        return self._initialized and self._llm is not None

    def startup(self) -> None:
        if self._initialized:
            return
        try:
            llm = AfriqueQwenLLM()
            llm.load_model()
            self._llm = llm
            self._memory.set_llm(llm)
            self._initialized = True
            self._init_error = None
            logger.info("MWIZERWA API: shared AfriqueQwen model loaded")
        except Exception as e:
            self._init_error = type(e).__name__
            logger.exception("MWIZERWA API: model initialization failed")
            raise

    def _new_engine(self) -> ConversationEngine:
        if self._llm is None:
            raise RuntimeError("Engine not initialized")
        manager = ConversationManager(llm=self._llm)
        return ConversationEngine(manager=manager, memory_coordinator=self._memory)

    def _resolve_session(self, session_id: Optional[str]) -> Tuple[str, ConversationEngine]:
        with self._sessions_lock:
            if session_id and session_id in self._sessions:
                return session_id, self._sessions[session_id]
            sid = session_id or str(uuid.uuid4())
            engine = self._new_engine()
            self._sessions[sid] = engine
            return sid, engine

    def process_chat(
        self,
        user_text: str,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[TurnContext] = None,
    ) -> Tuple[str, TurnResult]:
        if not self.engine_initialized:
            raise RuntimeError("Engine not initialized")

        conv_mode = ConversationMode(mode) if mode else None
        sid, engine = self._resolve_session(session_id)
        request = TurnRequest(
            user_text=user_text,
            mode=conv_mode,
            language=language,
            user_id=user_id,
            context=context,
        )

        with self._inference_lock:
            result = engine.process_turn(request)

        return sid, result

    def reset_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            engine = self._sessions.get(session_id)
            if engine is None:
                return False
            engine.reset_conversation()
            return True

    def session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    def get_tts(self) -> TTSEngine:
        with self._tts_lock:
            if self._tts is None:
                self._tts = TTSEngine()
                logger.info("MWIZERWA API: TTSEngine initialized (lazy)")
            return self._tts

    def get_asr(self) -> ASREngine:
        with self._asr_lock:
            if self._asr is None:
                self._asr = ASREngine()
                logger.info("MWIZERWA API: ASREngine initialized (lazy)")
            return self._asr

    def synthesize_tts(
        self,
        text: str,
        language: Optional[str] = None,
        speaker_id: int = 0,
    ) -> Path:
        lang = (language or "rw").strip().lower()
        tts = self.get_tts()
        model_status = tts.check_models(lang)
        if not model_status.get("present"):
            raise FileNotFoundError(
                model_status.get("instructions") or f"TTS model for '{lang}' is not available."
            )
        out_path = temp_output_wav()
        try:
            with self._tts_lock:
                wav = tts.synthesize_speech(
                    text,
                    output_path=out_path,
                    language=lang,
                    speaker_id=speaker_id,
                    check_lang=False if language else True,
                )
            if not wav or not Path(wav).is_file():
                safe_unlink(out_path)
                raise TTSSynthesisError("TTS produced no audio output.")
            return Path(wav)
        except (UnsupportedTTSLanguageError, ValueError) as e:
            safe_unlink(out_path)
            raise ValueError(str(e)) from e
        except Exception:
            safe_unlink(out_path)
            raise

    def transcribe_file(
        self, audio_path: Path, *, language: Optional[str] = None
    ) -> Tuple[str, str]:
        """Returns (transcript, effective ASR language code)."""
        effective_lang = resolve_asr_language(language)
        if effective_lang in ("en", "fr"):
            asr_lang: Optional[str] = effective_lang
        elif language is not None:
            asr_lang = effective_lang
        else:
            asr_lang = None
        with self._asr_lock:
            text = self.get_asr().transcribe_audio(audio_path, language=asr_lang)
        if not text.strip():
            raise ValueError("Transcription is empty.")
        return text.strip(), effective_lang

    @staticmethod
    def _safe_tts_client_message(exc: Exception) -> str:
        if isinstance(exc, FileNotFoundError):
            return "Speech synthesis models are not available on this server."
        if isinstance(exc, (UnsupportedTTSLanguageError, TTSSynthesisError, ValueError)):
            return "Speech synthesis could not be completed for this response."
        return "Speech synthesis failed."

    def process_voice_turn(
        self,
        audio_path: Path,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        language: Optional[str] = None,
        speaker_id: int = 0,
        context: Optional[TurnContext] = None,
    ) -> Tuple[str, str, str, TurnResult, Optional[bytes], Optional[str]]:
        """ASR → chat → TTS. Chat runs once; TTS failure does not re-run chat."""
        if not self.engine_initialized:
            raise RuntimeError("Engine not initialized")

        transcript, asr_lang = self.transcribe_file(audio_path, language=language)
        chat_lang: Optional[str] = None
        if language is not None and language.strip():
            chat_lang = resolve_asr_language(language)
        wav_path: Optional[Path] = None
        audio_bytes: Optional[bytes] = None
        audio_error: Optional[str] = None
        try:
            with self._inference_lock:
                sid, result = self.process_chat(
                    user_text=transcript,
                    session_id=session_id,
                    mode=mode,
                    language=chat_lang,
                    context=context,
                )
            response_lang = result.language
            try:
                wav_path = self.synthesize_tts(
                    result.response_text,
                    language=response_lang,
                    speaker_id=speaker_id,
                )
                audio_bytes = wav_path.read_bytes()
            except Exception as e:
                logger.exception("TTS failed after successful voice chat turn")
                audio_error = self._safe_tts_client_message(e)
            return sid, transcript, asr_lang, result, audio_bytes, audio_error
        finally:
            safe_unlink(wav_path)


_service: Optional[MWIZERWAApiService] = None
_service_lock = threading.Lock()


def get_api_service() -> MWIZERWAApiService:
    global _service
    with _service_lock:
        if _service is None:
            _service = MWIZERWAApiService()
        return _service