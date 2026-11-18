"""
MWIZERWA AI — Exercise Domain Unit Tests
Tests for Chunk 5 Phase 2A: Validates exercise categories, data models,
immutability, registry contents, language accessors, and safety-aligned content.
"""

import unittest
from dataclasses import FrozenInstanceError

from src.exercises import (
    Exercise,
    ExerciseCategory,
    ExerciseNotFoundError,
    ExerciseSessionState,
    ExerciseStatus,
    ExerciseStep,
    ExerciseTriggerType,
    ExerciseValidationError,
    SUPPORTED_EXERCISE_LANGUAGES,
    UnsupportedLanguageError,
    get_all_exercises,
    get_available_exercise_ids,
    get_exercise,
    get_exercises_by_category,
)


class TestExerciseEnums(unittest.TestCase):
    """Verifies enum values and definitions."""

    def test_exercise_categories_defined(self):
        expected_categories = {
            "breathing",
            "grounding",
            "thought_reflection",
            "self_compassion",
            "gratitude_journaling",
        }
        actual_categories = {cat.value for cat in ExerciseCategory}
        self.assertEqual(actual_categories, expected_categories)

    def test_exercise_statuses_defined(self):
        expected_statuses = {
            "idle",
            "offered",
            "in_progress",
            "completed",
            "declined",
            "abandoned",
        }
        actual_statuses = {status.value for status in ExerciseStatus}
        self.assertEqual(actual_statuses, expected_statuses)

    def test_trigger_types_defined(self):
        expected_triggers = {
            "explicit_user_request",
            "unsolicited_offer",
        }
        actual_triggers = {trig.value for trig in ExerciseTriggerType}
        self.assertEqual(actual_triggers, expected_triggers)


class TestExerciseStepModel(unittest.TestCase):
    """Verifies ExerciseStep immutability and validation."""

    def setUp(self):
        self.valid_step = ExerciseStep(
            step_number=1,
            instruction_rw="Humeka buhoro buhoro.",
            instruction_en="Breathe gently and slowly.",
            instruction_fr="Respirez doucement et lentement.",
            duration_seconds=15,
        )

    def test_step_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.valid_step.step_number = 2  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            self.valid_step.instruction_en = "New instruction"  # type: ignore

    def test_step_validation_invalid_step_number(self):
        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=0,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=-1,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=True,  # type: ignore
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=False,  # type: ignore
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
            )

    def test_step_validation_empty_instructions(self):
        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="",
                instruction_en="en",
                instruction_fr="fr",
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="rw",
                instruction_en="   ",
                instruction_fr="fr",
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="",
            )

    def test_step_validation_invalid_duration(self):
        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
                duration_seconds=0,
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
                duration_seconds=-5,
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
                duration_seconds=True,  # type: ignore
            )

        with self.assertRaises(ExerciseValidationError):
            ExerciseStep(
                step_number=1,
                instruction_rw="rw",
                instruction_en="en",
                instruction_fr="fr",
                duration_seconds=False,  # type: ignore
            )

    def test_step_language_accessor_and_no_silent_fallback(self):
        self.assertEqual(self.valid_step.get_instruction("rw"), "Humeka buhoro buhoro.")
        self.assertEqual(self.valid_step.get_instruction("en"), "Breathe gently and slowly.")
        self.assertEqual(self.valid_step.get_instruction("fr"), "Respirez doucement et lentement.")

        # Unsupported or missing language codes MUST raise UnsupportedLanguageError, NOT silently default
        with self.assertRaises(UnsupportedLanguageError):
            self.valid_step.get_instruction("es")

        with self.assertRaises(UnsupportedLanguageError):
            self.valid_step.get_instruction("de")

        with self.assertRaises(UnsupportedLanguageError):
            self.valid_step.get_instruction("")

        with self.assertRaises(UnsupportedLanguageError):
            self.valid_step.get_instruction(None)  # type: ignore


class TestExerciseModel(unittest.TestCase):
    """Verifies Exercise immutability and contract validation."""

    def setUp(self):
        self.steps = (
            ExerciseStep(
                step_number=1,
                instruction_rw="Intambwe ya mbere",
                instruction_en="Step one",
                instruction_fr="Étape un",
                duration_seconds=10,
            ),
            ExerciseStep(
                step_number=2,
                instruction_rw="Intambwe ya kabiri",
                instruction_en="Step two",
                instruction_fr="Étape deux",
                duration_seconds=15,
            ),
        )
        self.valid_exercise = Exercise(
            id="test_ex",
            category=ExerciseCategory.BREATHING,
            name_rw="Izina RW",
            name_en="Name EN",
            name_fr="Nom FR",
            description_rw="Ibisobanuro RW",
            description_en="Description EN",
            description_fr="Description FR",
            estimated_duration_minutes=2,
            steps=self.steps,
            cautionary_notes=("Icyitonderwa",),
        )

    def test_exercise_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.valid_exercise.id = "other_id"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            self.valid_exercise.name_en = "New Name"  # type: ignore

    def test_exercise_validation_empty_id(self):
        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=1,
                steps=self.steps,
            )

    def test_exercise_validation_invalid_category(self):
        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category="invalid_category",  # type: ignore
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=1,
                steps=self.steps,
            )

    def test_exercise_validation_empty_translations(self):
        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="   ",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=1,
                steps=self.steps,
            )

        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="",
                description_fr="fr",
                estimated_duration_minutes=1,
                steps=self.steps,
            )

    def test_exercise_validation_invalid_duration(self):
        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=0,
                steps=self.steps,
            )

        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=True,  # type: ignore
                steps=self.steps,
            )

        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=False,  # type: ignore
                steps=self.steps,
            )

    def test_exercise_validation_empty_steps(self):
        with self.assertRaises(ExerciseValidationError):
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=1,
                steps=(),
            )

    def test_exercise_validation_non_consecutive_step_numbers(self):
        bad_steps = (
            ExerciseStep(step_number=1, instruction_rw="a", instruction_en="b", instruction_fr="c"),
            ExerciseStep(step_number=3, instruction_rw="a", instruction_en="b", instruction_fr="c"),  # skipped 2
        )
        with self.assertRaises(ExerciseValidationError) as ctx:
            Exercise(
                id="test",
                category=ExerciseCategory.BREATHING,
                name_rw="rw",
                name_en="en",
                name_fr="fr",
                description_rw="rw",
                description_en="en",
                description_fr="fr",
                estimated_duration_minutes=1,
                steps=bad_steps,
            )
        self.assertIn("consecutively", str(ctx.exception))

    def test_exercise_language_accessors_and_no_silent_fallback(self):
        self.assertEqual(self.valid_exercise.get_name("rw"), "Izina RW")
        self.assertEqual(self.valid_exercise.get_name("en"), "Name EN")
        self.assertEqual(self.valid_exercise.get_name("fr"), "Nom FR")

        self.assertEqual(self.valid_exercise.get_description("rw"), "Ibisobanuro RW")
        self.assertEqual(self.valid_exercise.get_description("en"), "Description EN")
        self.assertEqual(self.valid_exercise.get_description("fr"), "Description FR")

        self.assertIn("Wumvise", self.valid_exercise.get_completion_question("rw"))
        self.assertIn("How did", self.valid_exercise.get_completion_question("en"))
        self.assertIn("Comment", self.valid_exercise.get_completion_question("fr"))

        # Explicit failure on unsupported language
        with self.assertRaises(UnsupportedLanguageError):
            self.valid_exercise.get_name("sw")

        with self.assertRaises(UnsupportedLanguageError):
            self.valid_exercise.get_description("es")

        with self.assertRaises(UnsupportedLanguageError):
            self.valid_exercise.get_completion_question("it")

    def test_get_step_by_index(self):
        self.assertEqual(self.valid_exercise.get_step(0).step_number, 1)
        self.assertEqual(self.valid_exercise.get_step(1).step_number, 2)
        with self.assertRaises(IndexError):
            self.valid_exercise.get_step(2)
        self.assertEqual(self.valid_exercise.total_steps, 2)


class TestExerciseSessionState(unittest.TestCase):
    """Verifies state machine transitions and controlled state mutations."""

    def setUp(self):
        self.state = ExerciseSessionState()

    def test_initial_state(self):
        self.assertEqual(self.state.status, ExerciseStatus.IDLE)
        self.assertIsNone(self.state.active_exercise_id)
        self.assertEqual(self.state.current_step_index, 0)
        self.assertEqual(self.state.cooldown_turns_remaining, 0)
        self.assertEqual(self.state.completed_exercise_ids, ())

    def test_cooldown_tracking_and_turn_decrement(self):
        self.state.apply_cooldown(5)
        self.assertEqual(self.state.cooldown_turns_remaining, 5)

        self.state.record_user_turn()
        self.assertEqual(self.state.cooldown_turns_remaining, 4)

        for _ in range(4):
            self.state.record_user_turn()
        self.assertEqual(self.state.cooldown_turns_remaining, 0)

        # Floor at zero
        self.state.record_user_turn()
        self.assertEqual(self.state.cooldown_turns_remaining, 0)

        # Boolean turns rejected explicitly
        with self.assertRaises(ExerciseValidationError):
            self.state.apply_cooldown(True)  # type: ignore

        with self.assertRaises(ExerciseValidationError):
            self.state.apply_cooldown(False)  # type: ignore

    def test_start_exercise_lifecycle(self):
        self.state.start_exercise("breathing_gentle_rhythm", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=4)
        self.assertEqual(self.state.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(self.state.active_exercise_id, "breathing_gentle_rhythm")
        self.assertEqual(self.state.current_step_index, 0)
        self.assertEqual(self.state.total_steps, 4)
        self.assertEqual(self.state.trigger_type, ExerciseTriggerType.EXPLICIT_USER_REQUEST)

        # Advance step
        idx1 = self.state.advance_step()
        self.assertEqual(idx1, 1)
        self.assertEqual(self.state.current_step_index, 1)

    def test_cannot_advance_step_when_not_in_progress(self):
        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step()

    def test_complete_exercise(self):
        self.state.start_exercise("breathing_gentle_rhythm", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=4)
        self.state.complete_exercise()

        self.assertEqual(self.state.status, ExerciseStatus.COMPLETED)
        self.assertIsNone(self.state.active_exercise_id)
        self.assertIsNone(self.state.total_steps)
        self.assertIn("breathing_gentle_rhythm", self.state.completed_exercise_ids)
        # Completing an exercise applies 5-turn cooldown
        self.assertEqual(self.state.cooldown_turns_remaining, 5)

    def test_decline_offer_applies_cooldown(self):
        self.state.offered_exercise_id = "grounding_54321"
        self.state.status = ExerciseStatus.OFFERED

        self.state.decline_offer()
        self.assertEqual(self.state.status, ExerciseStatus.DECLINED)
        self.assertIsNone(self.state.offered_exercise_id)
        self.assertEqual(self.state.cooldown_turns_remaining, 5)

    def test_abandon_exercise_applies_cooldown(self):
        self.state.start_exercise("grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=5)
        self.state.advance_step()

        self.state.abandon_exercise()
        self.assertEqual(self.state.status, ExerciseStatus.ABANDONED)
        self.assertIsNone(self.state.active_exercise_id)
        self.assertIsNone(self.state.total_steps)
        self.assertEqual(self.state.cooldown_turns_remaining, 5)

    def test_reset_to_idle(self):
        self.state.start_exercise("grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=5)
        self.state.reset_to_idle()
        self.assertEqual(self.state.status, ExerciseStatus.IDLE)
        self.assertIsNone(self.state.active_exercise_id)
        self.assertIsNone(self.state.total_steps)
        self.assertEqual(self.state.current_step_index, 0)

    def test_set_established_language(self):
        self.state.set_established_language("fr")
        self.assertEqual(self.state.established_language, "fr")

        with self.assertRaises(UnsupportedLanguageError):
            self.state.set_established_language("es")

    def test_start_exercise_rejects_when_already_in_progress(self):
        self.state.start_exercise("breathing_gentle_rhythm", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=4)
        self.state.advance_step()

        # Attempting to start another exercise while IN_PROGRESS must fail
        with self.assertRaises(ExerciseValidationError) as ctx:
            self.state.start_exercise("grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=5)
        self.assertIn("already in progress", str(ctx.exception))

        # Assert state remains completely unchanged
        self.assertEqual(self.state.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(self.state.active_exercise_id, "breathing_gentle_rhythm")
        self.assertEqual(self.state.current_step_index, 1)
        self.assertEqual(self.state.total_steps, 4)

    def test_complete_exercise_rejects_when_not_in_progress(self):
        # IDLE
        with self.assertRaises(ExerciseValidationError):
            self.state.complete_exercise()
        self.assertEqual(self.state.status, ExerciseStatus.IDLE)

        # OFFERED
        self.state.offer_exercise("grounding_54321")
        with self.assertRaises(ExerciseValidationError):
            self.state.complete_exercise()
        self.assertEqual(self.state.status, ExerciseStatus.OFFERED)
        self.assertEqual(self.state.offered_exercise_id, "grounding_54321")

        # IN_PROGRESS without active_exercise_id
        self.state.status = ExerciseStatus.IN_PROGRESS
        self.state.active_exercise_id = None
        with self.assertRaises(ExerciseValidationError):
            self.state.complete_exercise()

    def test_decline_offer_rejects_when_not_offered(self):
        # IDLE
        with self.assertRaises(ExerciseValidationError):
            self.state.decline_offer()
        self.assertEqual(self.state.status, ExerciseStatus.IDLE)
        self.assertEqual(self.state.cooldown_turns_remaining, 0)

        # IN_PROGRESS
        self.state.start_exercise("breathing_gentle_rhythm", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=4)
        with self.assertRaises(ExerciseValidationError):
            self.state.decline_offer()
        self.assertEqual(self.state.status, ExerciseStatus.IN_PROGRESS)
        self.assertEqual(self.state.active_exercise_id, "breathing_gentle_rhythm")

        # OFFERED without offered_exercise_id
        self.state.reset_to_idle()
        self.state.status = ExerciseStatus.OFFERED
        self.state.offered_exercise_id = None
        with self.assertRaises(ExerciseValidationError):
            self.state.decline_offer()

    def test_abandon_exercise_rejects_when_not_in_progress(self):
        # IDLE
        with self.assertRaises(ExerciseValidationError):
            self.state.abandon_exercise()
        self.assertEqual(self.state.status, ExerciseStatus.IDLE)
        self.assertEqual(self.state.cooldown_turns_remaining, 0)

        # OFFERED
        self.state.offer_exercise("grounding_54321")
        with self.assertRaises(ExerciseValidationError):
            self.state.abandon_exercise()
        self.assertEqual(self.state.status, ExerciseStatus.OFFERED)
        self.assertEqual(self.state.cooldown_turns_remaining, 0)

        # IN_PROGRESS without active_exercise_id
        self.state.reset_to_idle()
        self.state.status = ExerciseStatus.IN_PROGRESS
        self.state.active_exercise_id = None
        with self.assertRaises(ExerciseValidationError):
            self.state.abandon_exercise()

    def test_advance_step_bounds_and_rejections(self):
        # 3-step exercise (valid indices: 0, 1, 2)
        self.state.start_exercise("test_ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=3)
        self.assertEqual(self.state.current_step_index, 0)

        # Advance to step 1
        idx1 = self.state.advance_step()
        self.assertEqual(idx1, 1)
        self.assertEqual(self.state.current_step_index, 1)

        # Advance to step 2 (final step)
        idx2 = self.state.advance_step()
        self.assertEqual(idx2, 2)
        self.assertEqual(self.state.current_step_index, 2)

        # Attempt to advance beyond step 2 must fail
        with self.assertRaises(ExerciseValidationError) as ctx:
            self.state.advance_step()
        self.assertIn("already at final step index", str(ctx.exception))

        # Assert index was not mutated
        self.assertEqual(self.state.current_step_index, 2)

        # Single step exercise (total_steps=1): cannot advance past index 0
        self.state.reset_to_idle()
        self.state.start_exercise("single_step", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=1)
        self.assertEqual(self.state.current_step_index, 0)
        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step()
        self.assertEqual(self.state.current_step_index, 0)

        # Supplying a larger per-call bound cannot bypass the stored bound
        self.state.reset_to_idle()
        self.state.start_exercise("constrained", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=3)
        self.assertEqual(self.state.advance_step(max_steps=2), 1)

        with self.assertRaises(ExerciseValidationError) as ctx:
            self.state.advance_step(max_steps=10)
        self.assertIn("cannot exceed stored total_steps", str(ctx.exception))
        # Ensure session index remains unchanged after rejected call
        self.assertEqual(self.state.current_step_index, 1)

    def test_advance_step_invalid_bounds(self):
        # Missing total_steps parameter is rejected
        with self.assertRaises(TypeError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST)  # type: ignore

        # None total_steps is rejected
        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=None)  # type: ignore

        # Boolean total_steps (True, False) must be rejected explicitly
        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=True)  # type: ignore

        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=False)  # type: ignore

        # total_steps <= 0 rejected at start_exercise
        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=0)

        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=-2)

        # Non-integer types (float, string) rejected
        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=2.5)  # type: ignore

        with self.assertRaises(ExerciseValidationError):
            self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps="3")  # type: ignore

        # Assert session state remains unchanged (still IDLE) after failed starts
        self.assertEqual(self.state.status, ExerciseStatus.IDLE)
        self.assertIsNone(self.state.active_exercise_id)
        self.assertIsNone(self.state.total_steps)

        # max_steps validation at advance_step
        self.state.start_exercise("ex", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=3)

        # Boolean max_steps (True, False) must be rejected explicitly
        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step(max_steps=True)  # type: ignore

        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step(max_steps=False)  # type: ignore

        # max_steps <= 0 rejected at advance_step
        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step(max_steps=0)
        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step(max_steps=-1)

        # Non-integer types (float, string) rejected
        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step(max_steps=1.5)  # type: ignore

        with self.assertRaises(ExerciseValidationError):
            self.state.advance_step(max_steps="2")  # type: ignore

        # Ensure index remains 0 after all rejected advance_step calls
        self.assertEqual(self.state.current_step_index, 0)

    def test_total_steps_cleared_on_lifecycle_completion_or_reset(self):
        self.state.start_exercise("breathing_gentle_rhythm", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=4)
        self.assertEqual(self.state.total_steps, 4)

        self.state.complete_exercise()
        self.assertIsNone(self.state.total_steps)

        self.state.start_exercise("grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=5)
        self.assertEqual(self.state.total_steps, 5)

        self.state.abandon_exercise()
        self.assertIsNone(self.state.total_steps)

        self.state.start_exercise("grounding_54321", ExerciseTriggerType.EXPLICIT_USER_REQUEST, total_steps=5)
        self.state.reset_to_idle()
        self.assertIsNone(self.state.total_steps)


class TestExerciseRegistry(unittest.TestCase):
    """Verifies static exercise registry completeness, uniqueness, and safety content."""

    def test_registry_contains_exactly_five_exercises(self):
        exercises = get_all_exercises()
        self.assertEqual(len(exercises), 5)
        exercise_ids = get_available_exercise_ids()
        self.assertEqual(len(exercise_ids), 5)
        self.assertEqual(len(set(exercise_ids)), 5)

    def test_all_five_categories_represented(self):
        categories_in_registry = {ex.category for ex in get_all_exercises()}
        self.assertEqual(categories_in_registry, set(ExerciseCategory))

        for cat in ExerciseCategory:
            cat_exercises = get_exercises_by_category(cat)
            self.assertGreaterEqual(len(cat_exercises), 1)

    def test_deterministic_lookup_and_unknown_id(self):
        ex = get_exercise("breathing_gentle_rhythm")
        self.assertEqual(ex.id, "breathing_gentle_rhythm")
        self.assertEqual(ex.category, ExerciseCategory.BREATHING)

        with self.assertRaises(ExerciseNotFoundError):
            get_exercise("non_existent_exercise")

        with self.assertRaises(ExerciseNotFoundError):
            get_exercise("")

    def test_all_exercises_have_complete_trilingual_content(self):
        for ex in get_all_exercises():
            for lang in SUPPORTED_EXERCISE_LANGUAGES:
                name = ex.get_name(lang)
                self.assertTrue(name and name.strip(), f"Missing {lang} name in {ex.id}")
                desc = ex.get_description(lang)
                self.assertTrue(desc and desc.strip(), f"Missing {lang} description in {ex.id}")
                q = ex.get_completion_question(lang)
                self.assertTrue(q and q.strip(), f"Missing {lang} completion question in {ex.id}")

            self.assertGreaterEqual(ex.estimated_duration_minutes, 1)
            self.assertGreaterEqual(len(ex.steps), 1)

            # Step validation
            for i, step in enumerate(ex.steps):
                self.assertEqual(step.step_number, i + 1)
                for lang in SUPPORTED_EXERCISE_LANGUAGES:
                    instruction = step.get_instruction(lang)
                    self.assertTrue(
                        instruction and instruction.strip(),
                        f"Missing {lang} instruction in {ex.id} step {step.step_number}",
                    )

    # -----------------------------------------------------------------------
    # Safety & Non-Clinical Content Verifications
    # -----------------------------------------------------------------------

    def test_breathing_exercise_safety_content(self):
        """
        Breathing must:
        - Use comfortable, unforced breathing.
        - Advise user to stop and return to natural breathing if dizzy or uncomfortable.
        - Not mandate aggressive or prolonged breath-holding.
        """
        breathing = get_exercise("breathing_gentle_rhythm")

        # Cautionary notes contain dizziness / comfort stop instructions
        notes_text = " ".join(breathing.cautionary_notes).lower()
        self.assertIn("dizzy", notes_text)
        self.assertIn("uzungerezwa", notes_text)
        self.assertIn("vertiges", notes_text)

        # Step 1 explicitly contains comfort / stop option
        step1_en = breathing.get_step(0).get_instruction("en").lower()
        self.assertIn("stop immediately", step1_en)
        self.assertIn("natural breathing", step1_en)

        step1_rw = breathing.get_step(0).get_instruction("rw").lower()
        self.assertIn("hagarika", step1_rw)
        self.assertIn("bisanzwe", step1_rw)

        # Step 2 breathing is gentle and unforced
        step2_en = breathing.get_step(1).get_instruction("en").lower()
        self.assertIn("without forcing", step2_en)

    def test_grounding_exercise_safety_content(self):
        """
        Grounding must:
        - Allow users to skip or adapt senses.
        - Not require physical movement.
        """
        grounding = get_exercise("grounding_54321")

        notes_text = " ".join(grounding.cautionary_notes).lower()
        self.assertIn("adapt or skip", notes_text)
        self.assertIn("gusimbuka cyangwa guhindura", notes_text)
        self.assertIn("adapter ou ignorer", notes_text)

        # Instructions explicitly tell user they can skip or imagine
        step1_en = grounding.get_step(0).get_instruction("en").lower()
        self.assertIn("skip to senses", step1_en)

    def test_thought_reflection_safety_content(self):
        """
        Thought reflection must:
        - Use gentle questions.
        - Avoid confrontational disputation or calling thoughts irrational.
        """
        reflection = get_exercise("reflection_gentle_perspective")

        desc_en = reflection.get_description("en").lower()
        self.assertIn("non-judgmental", desc_en)

        step1_en = reflection.get_step(0).get_instruction("en").lower()
        self.assertIn("without judging yourself", step1_en)

        step2_en = reflection.get_step(1).get_instruction("en").lower()
        self.assertIn("trusted friend", step2_en)

    def test_self_compassion_safety_content(self):
        """
        Self-compassion must:
        - Avoid forcing positive affirmations.
        - Acknowledge difficulty without requiring feeling differently.
        """
        compassion = get_exercise("compassion_pause")

        desc_en = compassion.get_description("en").lower()
        self.assertIn("without forcing positivity", desc_en)

        step3_en = compassion.get_step(2).get_instruction("en").lower()
        self.assertIn("without expecting yourself to suddenly feel happy", step3_en)

    def test_gratitude_safety_content(self):
        """
        Gratitude must:
        - Focus on realistic, concrete sources of comfort or stability.
        - Never imply gratitude for painful or abusive circumstances.
        """
        gratitude = get_exercise("gratitude_grounded_anchor")

        notes_text = " ".join(gratitude.cautionary_notes).lower()
        self.assertIn("never asked to feel grateful for painful", notes_text)
        self.assertIn("ntusabwa gushimira ibintu bibabaje", notes_text)

        step1_en = gratitude.get_step(0).get_instruction("en").lower()
        self.assertIn("does not require that the day was good", step1_en)

    def test_registry_mapping_is_immutable_proxy(self):
        """Verifies that internal registry mapping is a read-only MappingProxyType."""
        from types import MappingProxyType
        from src.exercises.registry import _EXERCISES

        self.assertIsInstance(_EXERCISES, MappingProxyType)

        # Attempting mutation must fail
        with self.assertRaises(TypeError):
            _EXERCISES["new_key"] = None  # type: ignore

        with self.assertRaises(TypeError):
            del _EXERCISES["breathing_gentle_rhythm"]  # type: ignore


if __name__ == "__main__":
    unittest.main()
