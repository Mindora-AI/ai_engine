"""
MWIZERWA AI — Exercise Domain Models
Defines immutable exercise definitions, steps, session state, and language validation contracts.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence

from src.exercises.categories import ExerciseCategory, ExerciseStatus, ExerciseTriggerType

SUPPORTED_EXERCISE_LANGUAGES: Tuple[str, ...] = ("rw", "en", "fr")


class UnsupportedLanguageError(ValueError):
    """Raised when an exercise accessor is called with an unsupported or missing language code."""
    pass


class ExerciseValidationError(ValueError):
    """Raised when an exercise or exercise step fails invariant or domain contract validation."""
    pass


@dataclass(frozen=True)
class ExerciseStep:
    """
    Immutable representation of an individual step within an exercise.
    Each step must have verified, non-empty instructions in Kinyarwanda, English, and French.
    """
    step_number: int
    instruction_rw: str
    instruction_en: str
    instruction_fr: str
    duration_seconds: Optional[int] = None
    allow_skip: bool = True

    def __post_init__(self):
        if isinstance(self.step_number, bool) or not isinstance(self.step_number, int) or self.step_number < 1:
            raise ExerciseValidationError(
                f"step_number must be a positive integer >= 1, got {self.step_number}."
            )

        for lang, text in (
            ("rw", self.instruction_rw),
            ("en", self.instruction_en),
            ("fr", self.instruction_fr),
        ):
            if not isinstance(text, str) or not text.strip():
                raise ExerciseValidationError(
                    f"instruction_{lang} must be a non-empty string."
                )

        if self.duration_seconds is not None:
            if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, int) or self.duration_seconds < 1:
                raise ExerciseValidationError(
                    f"duration_seconds must be a positive integer >= 1 when provided, got {self.duration_seconds}."
                )

    def get_instruction(self, language: str) -> str:
        """
        Retrieves the step instruction in the requested language.

        Raises:
            UnsupportedLanguageError: If the language is not 'rw', 'en', or 'fr'.
        """
        norm = (language or "").strip().lower()
        if norm not in SUPPORTED_EXERCISE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Unsupported exercise language: '{language}'. "
                f"Supported languages are: {SUPPORTED_EXERCISE_LANGUAGES}."
            )
        if norm == "rw":
            return self.instruction_rw
        elif norm == "en":
            return self.instruction_en
        else:
            return self.instruction_fr


@dataclass(frozen=True)
class Exercise:
    """
    Immutable representation of a verified wellness exercise.
    Contains complete trilingual descriptions, ordered steps, and cautionary notes.
    """
    id: str
    category: ExerciseCategory
    name_rw: str
    name_en: str
    name_fr: str
    description_rw: str
    description_en: str
    description_fr: str
    estimated_duration_minutes: int
    steps: Tuple[ExerciseStep, ...]
    cautionary_notes: Tuple[str, ...] = ()
    supported_languages: Tuple[str, ...] = SUPPORTED_EXERCISE_LANGUAGES
    completion_question_rw: str = "Wumvise bimeze bite nyuma yo kubikora?"
    completion_question_en: str = "How did that feel for you?"
    completion_question_fr: str = "Comment vous êtes-vous senti en faisant cela ?"

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ExerciseValidationError("Exercise id must be a non-empty string.")

        if not isinstance(self.category, ExerciseCategory):
            raise ExerciseValidationError(
                f"category must be an ExerciseCategory enum, got {self.category}."
            )

        for lang, text in (
            ("rw", self.name_rw),
            ("en", self.name_en),
            ("fr", self.name_fr),
        ):
            if not isinstance(text, str) or not text.strip():
                raise ExerciseValidationError(f"name_{lang} must be a non-empty string.")

        for lang, text in (
            ("rw", self.description_rw),
            ("en", self.description_en),
            ("fr", self.description_fr),
        ):
            if not isinstance(text, str) or not text.strip():
                raise ExerciseValidationError(f"description_{lang} must be a non-empty string.")

        if isinstance(self.estimated_duration_minutes, bool) or not isinstance(self.estimated_duration_minutes, int) or self.estimated_duration_minutes < 1:
            raise ExerciseValidationError(
                f"estimated_duration_minutes must be an integer >= 1, got {self.estimated_duration_minutes}."
            )

        # Validate and enforce steps as immutable tuple
        if not isinstance(self.steps, Sequence) or len(self.steps) == 0:
            raise ExerciseValidationError("Exercise must contain at least one ExerciseStep.")

        steps_tuple = tuple(self.steps)
        object.__setattr__(self, "steps", steps_tuple)

        for i, step in enumerate(steps_tuple):
            if not isinstance(step, ExerciseStep):
                raise ExerciseValidationError(
                    f"All items in steps must be ExerciseStep instances, got {type(step).__name__} at index {i}."
                )
            expected_num = i + 1
            if step.step_number != expected_num:
                raise ExerciseValidationError(
                    f"Steps must be numbered consecutively starting from 1. "
                    f"Expected step_number {expected_num} at index {i}, got {step.step_number}."
                )

        # Enforce cautionary notes as tuple
        if not isinstance(self.cautionary_notes, tuple):
            object.__setattr__(self, "cautionary_notes", tuple(self.cautionary_notes))

        for lang, text in (
            ("rw", self.completion_question_rw),
            ("en", self.completion_question_en),
            ("fr", self.completion_question_fr),
        ):
            if not isinstance(text, str) or not text.strip():
                raise ExerciseValidationError(
                    f"completion_question_{lang} must be a non-empty string."
                )

    def get_name(self, language: str) -> str:
        """Retrieves exercise name in the requested language."""
        norm = (language or "").strip().lower()
        if norm not in SUPPORTED_EXERCISE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Unsupported exercise language: '{language}'. Supported: {SUPPORTED_EXERCISE_LANGUAGES}."
            )
        if norm == "rw":
            return self.name_rw
        elif norm == "en":
            return self.name_en
        else:
            return self.name_fr

    def get_description(self, language: str) -> str:
        """Retrieves exercise description in the requested language."""
        norm = (language or "").strip().lower()
        if norm not in SUPPORTED_EXERCISE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Unsupported exercise language: '{language}'. Supported: {SUPPORTED_EXERCISE_LANGUAGES}."
            )
        if norm == "rw":
            return self.description_rw
        elif norm == "en":
            return self.description_en
        else:
            return self.description_fr

    def get_completion_question(self, language: str) -> str:
        """Retrieves exercise completion check-in question in the requested language."""
        norm = (language or "").strip().lower()
        if norm not in SUPPORTED_EXERCISE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Unsupported exercise language: '{language}'. Supported: {SUPPORTED_EXERCISE_LANGUAGES}."
            )
        if norm == "rw":
            return self.completion_question_rw
        elif norm == "en":
            return self.completion_question_en
        else:
            return self.completion_question_fr

    def get_step(self, step_index: int) -> ExerciseStep:
        """
        Retrieves step by 0-based index.

        Raises:
            IndexError: If step_index is out of range.
        """
        if not (0 <= step_index < len(self.steps)):
            raise IndexError(
                f"Step index {step_index} out of range (0..{len(self.steps) - 1})."
            )
        return self.steps[step_index]

    @property
    def total_steps(self) -> int:
        """Returns total number of steps in the exercise."""
        return len(self.steps)


@dataclass
class ExerciseSessionState:
    """
    Mutable state tracker for active, offered, and completed exercises within a session.
    Provides controlled mutation methods enforcing lifecycle invariants.
    """
    status: ExerciseStatus = ExerciseStatus.IDLE
    active_exercise_id: Optional[str] = None
    current_step_index: int = 0
    total_steps: Optional[int] = None
    trigger_type: Optional[ExerciseTriggerType] = None
    offered_exercise_id: Optional[str] = None
    cooldown_turns_remaining: int = 0
    completed_exercise_ids: Tuple[str, ...] = ()
    established_language: Optional[str] = None

    def record_user_turn(self) -> None:
        """
        Decrements the active cooldown counter by 1 on each non-empty user turn,
        flooring at 0.
        """
        if self.cooldown_turns_remaining > 0:
            self.cooldown_turns_remaining -= 1

    def apply_cooldown(self, turns: int = 5) -> None:
        """
        Sets the unsolicited recommendation cooldown in turns (default 5).
        """
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
            raise ExerciseValidationError(f"Cooldown turns must be a non-negative integer, got {turns}.")
        self.cooldown_turns_remaining = max(self.cooldown_turns_remaining, turns)

    def offer_exercise(self, exercise_id: str) -> None:
        """
        Transitions session state to OFFERED for the given exercise_id.
        """
        if self.status == ExerciseStatus.IN_PROGRESS:
            raise ExerciseValidationError(
                f"Cannot offer exercise '{exercise_id}': an exercise is already in progress."
            )
        if not isinstance(exercise_id, str) or not exercise_id.strip():
            raise ExerciseValidationError("exercise_id must be a non-empty string.")
        self.status = ExerciseStatus.OFFERED
        self.offered_exercise_id = exercise_id.strip()

    def start_exercise(
        self,
        exercise_id: str,
        trigger: ExerciseTriggerType,
        total_steps: int,
    ) -> None:
        """
        Transitions session state to IN_PROGRESS for the given exercise_id.

        Raises:
            ExerciseValidationError: If an exercise is already in progress,
                                      if exercise_id or trigger is invalid,
                                      or if total_steps is not an integer >= 1.
        """
        if self.status == ExerciseStatus.IN_PROGRESS:
            raise ExerciseValidationError(
                f"Cannot start exercise '{exercise_id}': another exercise ('{self.active_exercise_id}') "
                f"is already in progress."
            )
        if not isinstance(exercise_id, str) or not exercise_id.strip():
            raise ExerciseValidationError("exercise_id must be a non-empty string.")
        if not isinstance(trigger, ExerciseTriggerType):
            raise ExerciseValidationError(f"trigger must be an ExerciseTriggerType, got {trigger}.")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
            raise ExerciseValidationError(
                f"total_steps must be a positive integer >= 1, got {total_steps}."
            )

        self.status = ExerciseStatus.IN_PROGRESS
        self.active_exercise_id = exercise_id.strip()
        self.current_step_index = 0
        self.total_steps = total_steps
        self.trigger_type = trigger
        self.offered_exercise_id = None

    def advance_step(self, max_steps: Optional[int] = None) -> int:
        """
        Advances the current step index by 1 and returns the new index.

        Raises:
            ExerciseValidationError: If exercise is not in progress,
                                      if session has invalid total_steps,
                                      if max_steps exceeds stored total_steps,
                                      or if advancing would exceed the step bound.
        """
        if self.status != ExerciseStatus.IN_PROGRESS:
            raise ExerciseValidationError(
                f"Cannot advance step when exercise is not in progress (current status: {self.status})."
            )

        if self.total_steps is None or self.total_steps < 1:
            raise ExerciseValidationError(
                f"Invalid session state: active exercise has invalid total_steps ({self.total_steps})."
            )

        bound = self.total_steps
        if max_steps is not None:
            if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
                raise ExerciseValidationError(
                    f"max_steps must be a positive integer >= 1, got {max_steps}."
                )
            if max_steps > self.total_steps:
                raise ExerciseValidationError(
                    f"per-call max_steps ({max_steps}) cannot exceed stored total_steps ({self.total_steps})."
                )
            bound = min(bound, max_steps)

        if self.current_step_index + 1 >= bound:
            raise ExerciseValidationError(
                f"Cannot advance step: already at final step index {self.current_step_index} "
                f"(total steps: {bound})."
            )

        self.current_step_index += 1
        return self.current_step_index

    def complete_exercise(self) -> None:
        """
        Transitions session state to COMPLETED, records completed exercise id,
        resets active state and step bound, and applies the standard 5-turn unsolicited cooldown.

        Raises:
            ExerciseValidationError: If session is not IN_PROGRESS or no active exercise is set.
        """
        if self.status != ExerciseStatus.IN_PROGRESS:
            raise ExerciseValidationError(
                f"Cannot complete exercise: session status is {self.status}, expected {ExerciseStatus.IN_PROGRESS}."
            )
        if not self.active_exercise_id:
            raise ExerciseValidationError(
                "Cannot complete exercise: no active_exercise_id is set."
            )

        if self.active_exercise_id not in self.completed_exercise_ids:
            self.completed_exercise_ids = self.completed_exercise_ids + (self.active_exercise_id,)

        self.status = ExerciseStatus.COMPLETED
        self.active_exercise_id = None
        self.current_step_index = 0
        self.total_steps = None
        self.apply_cooldown(5)

    def decline_offer(self) -> None:
        """
        Records that the user declined an offered exercise and applies the 5-turn cooldown.

        Raises:
            ExerciseValidationError: If session is not in OFFERED status or no offered exercise is set.
        """
        if self.status != ExerciseStatus.OFFERED:
            raise ExerciseValidationError(
                f"Cannot decline exercise offer: session status is {self.status}, expected {ExerciseStatus.OFFERED}."
            )
        if not self.offered_exercise_id:
            raise ExerciseValidationError(
                "Cannot decline exercise offer: no offered_exercise_id is set."
            )

        self.status = ExerciseStatus.DECLINED
        self.offered_exercise_id = None
        self.active_exercise_id = None
        self.current_step_index = 0
        self.total_steps = None
        self.apply_cooldown(5)

    def abandon_exercise(self) -> None:
        """
        Records that an active exercise was stopped/abandoned early by the user
        and applies the 5-turn cooldown.

        Raises:
            ExerciseValidationError: If session is not IN_PROGRESS or no active exercise is set.
        """
        if self.status != ExerciseStatus.IN_PROGRESS:
            raise ExerciseValidationError(
                f"Cannot abandon exercise: session status is {self.status}, expected {ExerciseStatus.IN_PROGRESS}."
            )
        if not self.active_exercise_id:
            raise ExerciseValidationError(
                "Cannot abandon exercise: no active_exercise_id is set."
            )

        self.status = ExerciseStatus.ABANDONED
        self.active_exercise_id = None
        self.current_step_index = 0
        self.total_steps = None
        self.apply_cooldown(5)

    def reset_to_idle(self) -> None:
        """
        Resets the active exercise to IDLE state without affecting cooldown or completed history.
        """
        self.status = ExerciseStatus.IDLE
        self.active_exercise_id = None
        self.current_step_index = 0
        self.total_steps = None
        self.offered_exercise_id = None
        self.trigger_type = None

    def set_established_language(self, language: str) -> None:
        """
        Records the verified, established language of the session.
        """
        norm = (language or "").strip().lower()
        if norm not in SUPPORTED_EXERCISE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Cannot set established language to unsupported '{language}'. "
                f"Supported languages are: {SUPPORTED_EXERCISE_LANGUAGES}."
            )
        self.established_language = norm
