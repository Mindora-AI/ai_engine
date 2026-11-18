"""
MWIZERWA AI — Exercise Lifecycle Management
Coordinates the active lifecycle of wellness exercises:
- Delivering deterministic static instructions in pre-defined step order.
- Processing user control inputs (continue, stop, decline, answer, unrelated).
- Enforcing state transitions and completion check-ins across Kinyarwanda, English, and French.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from src.exercises.categories import ExerciseStatus, ExerciseTriggerType
from src.exercises.models import (
    Exercise,
    ExerciseSessionState,
    ExerciseStep,
    ExerciseValidationError,
    SUPPORTED_EXERCISE_LANGUAGES,
    UnsupportedLanguageError,
)
from src.exercises.registry import get_exercise, ExerciseNotFoundError


class UserResponseType(str, Enum):
    """Classification of user input during an offered or active exercise."""
    ACCEPT = "accept"
    DECLINE = "decline"
    CONTINUE = "continue"
    STOP = "stop"
    ANSWER = "answer"
    UNRELATED = "unrelated"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class LifecycleResult:
    """Outcome of processing an exercise turn in the lifecycle manager."""
    response_text: str
    status: ExerciseStatus
    exercise_id: Optional[str] = None
    step_number: Optional[int] = None
    total_steps: Optional[int] = None
    is_completed: bool = False
    is_stopped: bool = False
    is_offered: bool = False


# Pre-authored localized conversational phrasing for lifecycle events

STOP_ACKNOWLEDGMENTS = {
    "rw": (
        "Nibyo rwose, nta kibazo na gato. Ushobora guhagarika igihe cyose ubishakiye. "
        "Ndi hano kugira ngo ngutege amatwi; wumva umeze ute muri aka kanya?"
    ),
    "en": (
        "That is completely okay. You can stop at any time. "
        "I am right here with you; how are you feeling right now?"
    ),
    "fr": (
        "C'est tout à fait normal et compréhensible. Vous pouvez vous arrêter quand vous le souhaitez. "
        "Je reste à votre écoute ; comment vous sentez-vous en ce moment ?"
    ),
}

DECLINE_ACKNOWLEDGMENTS = {
    "rw": (
        "Nta kibazo na gato, turakomeza ikiganiro cyacu bisanzwe. Mbwira ibikuraje ishinga."
    ),
    "en": (
        "That is completely fine. We will just keep talking naturally. Tell me what is on your mind."
    ),
    "fr": (
        "Il n'y a aucun souci, nous poursuivons notre échange tout simplement. Dites-moi ce qui vous préoccupe."
    ),
}

UNRELATED_CHECKINS = {
    "rw": (
        "Ndumva ibyo umbwiye. Ese wifuza ko duhagarika uyu mwitozo tugakomeza kubiganiraho, "
        "cyangwa urifuza ko dukomeza intambwe ikurikira?"
    ),
    "en": (
        "I hear what you are saying. Would you like to pause our exercise so we can talk about this instead, "
        "or would you like to continue to the next step?"
    ),
    "fr": (
        "Je vous entends bien. Souhaitez-vous arrêter cet exercice pour que nous parlions de cela, "
        "ou préférez-vous continuer vers l'étape suivante ?"
    ),
}

OFFER_TEMPLATES = {
    "rw": (
        "Niba ubyifuza, dushobora gufata umwanya muto tugakora umwitozo woroheje witwa '{name}' "
        "(umara iminota hafi {duration}). Ese wifuza ko tuwukorana, cyangwa urifuza ko dukomeza kwiganirira?"
    ),
    "en": (
        "If you would like, we could take a gentle pause to try a short exercise called '{name}' "
        "(about {duration} minutes). Would you like to do that together, or would you prefer to keep talking?"
    ),
    "fr": (
        "Si vous le souhaitez, nous pourrions faire une courte pause pour essayer l'exercice '{name}' "
        "(environ {duration} minutes). Aimeriez-vous que nous le fassions ensemble, ou préférez-vous continuer à échanger ?"
    ),
}


class ExerciseLifecycle:
    """
    Manages deterministic exercise delivery, state machine progression,
    and user control command parsing.
    """

    def __init__(self):
        self._compile_control_patterns()

    def _compile_control_patterns(self) -> None:
        """Compiles regex patterns for user control interpretation."""

        # Stop / Cancel requests (Highest control priority)
        self._stop_pattern = re.compile(
            r"\b((?:stop(?: it| the exercise)?|i want to stop|i don'?t want to continue|i do not want to continue|cancel|quit|abort|end this|"
            r"i('?m| am) done with this|no more|leave this)|"
            r"(?:hagarika|reka kubikora|sinshaka gukomeza|ntabwo nshaka gukomeza|ndahagaritse|reka tubireke|bireke|ndarambiwe)|"
            r"(?:arrête|arrêter|je veux arrêter|je ne veux pas continuer|annuler|stop|quitter|c'est bon j'arrête|on arrête))\b",
            re.IGNORECASE,
        )

        # Hesitation / Negation of readiness / Requests to pause or wait (Must return AMBIGUOUS)
        self._hesitation_pattern = re.compile(
            r"\b((?:i('?m| am) not (?:okay|ok|ready(?: yet)?)|not (?:ready(?: yet)?|yet|now|okay|ok)|(?:just )?not yet|don'?t feel ready|pause|wait|hold on|"
            r"(?:give me|need|take) a (?:moment|minute|second)|(?:want to continue|like to continue),? (?:just )?not yet)|"
            r"(?:ntabwo niteguye|siniteguye|ntabwo bimeze neza|reka da|tegerezaho?|ba uretse|mpa akanya|akanya gato|(?:ariko )?si ubu|ndashaka gukomeza,? ariko si ubu)|"
            r"(?:je ne suis pas prêt|pas prêt(?: encore)?|pas encore|pas d'accord|pas maintenant|(?:mais )?pas maintenant|attends|attendez(?: un moment)?|(?:attendez|laissez-moi|donnez-moi|juste|besoin d') un (?:moment|instant)|un instant))\b",
            re.IGNORECASE,
        )

        # Standalone affirmative acknowledgments authorizing continuation
        self._standalone_ack = re.compile(
            r"^(?:ok|okay|yes|yeah|sure|yego|sawa|birakozwe|oui|d'accord|done|finished)[.!]?$",
            re.IGNORECASE,
        )

        # Explicit continuation commands and readiness declarations
        self._continue_pattern = re.compile(
            r"\b((?:next(?: step(?: please)?)?|continue|proceed|go ahead|let'?s continue|let'?s move on)|"
            r"(?:(?:i('?m| am) )?ready(?: now| for next| to continue)?|ready for the next step)|"
            r"(?:done with (?:that|this|it)(?: step)?|finished with (?:that|this|it)(?: step)?|all done)|"
            r"(?:komeza(?: imbere)?|nditeguye|nsoje(?: iyi ntambwe)?|ndabyumvise|intambwe ikurikira|dufite iki gikurikira)|"
            r"(?:suivant|continue|continuer|(?:je suis )?prêt|c'est fait|j'ai fini(?: cette étape)?|on continue|étape suivante))\b",
            re.IGNORECASE,
        )

        # Acceptance of an offered exercise
        self._accept_pattern = re.compile(
            r"\b((?:yes|yeah|sure|okay|ok|let's do it|i'd like that|please|sounds good|yes please)|"
            r"(?:yego|yego ndabishaka|sawa|reka tubikore|mfasha|nta kibazo|tujyemo)|"
            r"(?:oui|d'accord|volontiers|pourquoi pas|je veux bien|allons-y|oui s'il vous plaît))\b",
            re.IGNORECASE,
        )

        # Decline / Refusal of an offered exercise
        self._decline_pattern = re.compile(
            r"\b((?:no|no thanks|not now|i don't want to|maybe later|i'd rather talk|just listen|not really)|"
            r"(?:oya|oya ntabwo nshaka|reka da|si ubu|nyuma|ndashaka kuvuga gusa|reka)|"
            r"(?:non|non merci|pas maintenant|je préfère parler|une autre fois|pas envie|non pas maintenant))\b",
            re.IGNORECASE,
        )

        # Unrelated Question Check (General questions about outside topics)
        self._unrelated_question = re.compile(
            r"(?:\?|"
            r"\b(?:what time|when (?:does|is|will|did)|why (?:did|didn'?t|does|is|are)|who (?:is|was)|where (?:is|are|can)|"
            r"can you (?:explain|tell me about|help with)|could you (?:explain|tell)|how much|how many)\b|"
            r"\b(?:kuki|kubera iki|ryari|hehe|nde|isaha ngaho|ushobora gusobanura)\b|"
            r"\b(?:quelle heure|pourquoi|quand|où est|qui est|pouvez-vous expliquer|peux-tu expliquer)\b)",
            re.IGNORECASE,
        )

        # Unrelated Departure / External statement check
        self._unrelated_statement = re.compile(
            r"\b((?:i (?:need|have) to (?:go|leave|run)|someone is at the door|i am busy|i have a meeting|"
            r"my (?:boss|coworker|car|phone|laptop|flight|bus|package)|"
            r"the (?:weather|store|shop|market|traffic|train|bus|game|movie|meeting|doctor))|"
            r"(?:ngomba kugenda|ndagiye|ndahuze|mfite inama|imodoka|isoko)|"
            r"(?:je dois (?:partir|y aller|m'en aller)|on m'appelle|je suis occupé|j'ai une réunion|"
            r"mon (?:patron|téléphone|ordinateur)|le (?:magasin|train|bus|médecin)))\b",
            re.IGNORECASE,
        )

        # Exercise/Step-relevant vocabulary across senses, breathing, thoughts, compassion, and gratitude
        self._step_relevant_pattern = re.compile(
            r"\b((?:see|saw|notice|looking|look|feel|felt|touch|hear|heard|listen|sound|smell|scent|taste|"
            r"breath|breathe|inhale|exhale|chest|shoulders|air|lungs|relax|sitting|sitting comfortably|"
            r"thought|worry|worried|thinking|perspective|situation|stress|pressure|mind|opportunity|learning|"
            r"hard|difficult|struggle|tough|heart|hand|kind|compassion|gentle|exhausting|alone|"
            r"grateful|gratitude|thankful|appreciate|glad|anchor|pen|table|chair|window|desk|shoes|floor|lamp|tea|safe place)|"
            r"(?:bona|nabonye|kora|reba|umva|numvise|amajwi|impumuro|uburyohe|"
            r"humeka|guhumeka|umwuka|ibitugu|ituze|gutuza|kwicara|niyicariye|"
            r"igitekerezo|ibitekerezo|impungenge|inshuti|kureba|gutekereza|gutsinda|ikizamini|"
            r"bikomeye|biragoye|umutima|kwiyitaho|impuhwe|birananiye|igituza|jyenyine|"
            r"ndashimira|gushimira|ibyiza|intebe|ameza|idirishya|igitabo|ikawa|ahantu hatekanye)|"
            r"(?:vois|vu|regarde|sens|ressens|touche|entends|entendu|écoute|odeur|goût|"
            r"respire|respirer|souffle|inspirer?|inspiré|inspiration|expirer?|expiré|expiration|épaules|poumons|calme|installé|"
            r"pensée|pensées|inquiétude|situation|ami|regard|esprit|projet|réussir|"
            r"difficile|dur|épreuve|cœur|main|bienveillance|éprouvant|seul|doux|"
            r"reconnaissant|gratitude|merci|apprécie|chaise|table|fenêtre|lampe|livre|café|endroit calme))\b",
            re.IGNORECASE,
        )

    def classify_user_response(
        self,
        user_text: str,
        current_status: ExerciseStatus,
        active_exercise_id: Optional[str] = None,
        step_index: Optional[int] = None,
    ) -> UserResponseType:
        """
        Classifies incoming user text relative to the current exercise lifecycle state.
        Distinguishes explicit continuation, affirmative acknowledgments, step-relevant answers,
        unrelated questions/statements, hesitation/negation, and ambiguous phrasing.
        """
        cleaned = (user_text or "").strip()
        if not cleaned:
            return UserResponseType.AMBIGUOUS

        # 1. Immediate Stop Check (Highest control priority)
        if self._stop_pattern.search(cleaned):
            return UserResponseType.STOP

        # 2. Offer Response Check
        if current_status == ExerciseStatus.OFFERED:
            if self._decline_pattern.search(cleaned):
                return UserResponseType.DECLINE
            if self._accept_pattern.search(cleaned):
                return UserResponseType.ACCEPT
            return UserResponseType.UNRELATED

        # 3. Active Exercise Progression Check
        if current_status == ExerciseStatus.IN_PROGRESS:
            # 3A. Negation / Hesitation check (Must NOT advance)
            if self._hesitation_pattern.search(cleaned):
                return UserResponseType.AMBIGUOUS

            # 3B. Explicit Continuation / Acknowledgments
            if self._standalone_ack.match(cleaned) or self._continue_pattern.search(cleaned):
                return UserResponseType.CONTINUE

            # 3C. Unrelated questions or departure/external statements
            if self._unrelated_question.search(cleaned) or self._unrelated_statement.search(cleaned):
                return UserResponseType.UNRELATED

            # 3D. Exercise / Step relevance check
            if self._step_relevant_pattern.search(cleaned):
                return UserResponseType.ANSWER

            # 3E. Non-relevant input with >= 3 words is UNRELATED (e.g. off-topic life chatter)
            word_count = len(cleaned.split())
            if word_count >= 3:
                return UserResponseType.UNRELATED

            # Short non-matching input (< 3 words) is AMBIGUOUS
            return UserResponseType.AMBIGUOUS

        return UserResponseType.UNRELATED

    def format_offer(self, exercise_id: str, language: str) -> str:
        """
        Generates deterministic conversational framing when offering an exercise.
        """
        norm_lang = (language or "rw").strip().lower()
        if norm_lang not in SUPPORTED_EXERCISE_LANGUAGES:
            norm_lang = "rw"

        exercise = get_exercise(exercise_id)
        name = exercise.get_name(norm_lang)
        duration = exercise.estimated_duration_minutes

        template = OFFER_TEMPLATES[norm_lang]
        return template.format(name=name, duration=duration)

    def offer_exercise(
        self,
        session_state: ExerciseSessionState,
        exercise_id: str,
        language: str,
    ) -> LifecycleResult:
        """
        Presents an optional exercise offer to the user.
        """
        exercise = get_exercise(exercise_id)
        session_state.offer_exercise(exercise.id)

        offer_text = self.format_offer(exercise.id, language)
        return LifecycleResult(
            response_text=offer_text,
            status=ExerciseStatus.OFFERED,
            exercise_id=exercise.id,
            total_steps=exercise.total_steps,
            is_offered=True,
        )

    def start_exercise(
        self,
        session_state: ExerciseSessionState,
        exercise_id: str,
        trigger: ExerciseTriggerType,
        language: str,
    ) -> LifecycleResult:
        """
        Initiates an exercise and delivers its first instruction.
        """
        exercise = get_exercise(exercise_id)
        session_state.start_exercise(exercise.id, trigger, total_steps=exercise.total_steps)

        norm_lang = (language or "rw").strip().lower()
        if norm_lang not in SUPPORTED_EXERCISE_LANGUAGES:
            norm_lang = "rw"

        first_step = exercise.get_step(0)
        instruction = first_step.get_instruction(norm_lang)

        # Build clean speakable step presentation
        step_header = f"Step 1 of {exercise.total_steps}: "
        if norm_lang == "rw":
            step_header = f"Intambwe ya 1 kuri {exercise.total_steps}: "
        elif norm_lang == "fr":
            step_header = f"Étape 1 sur {exercise.total_steps} : "

        response_text = f"{step_header}{instruction}"
        return LifecycleResult(
            response_text=response_text,
            status=ExerciseStatus.IN_PROGRESS,
            exercise_id=exercise.id,
            step_number=1,
            total_steps=exercise.total_steps,
        )

    def advance_exercise(
        self,
        session_state: ExerciseSessionState,
        language: str,
    ) -> LifecycleResult:
        """
        Advances to the next step of an active exercise or completes it.
        """
        if session_state.status != ExerciseStatus.IN_PROGRESS or not session_state.active_exercise_id:
            raise ExerciseValidationError(
                f"Cannot advance exercise: session status is {session_state.status}."
            )

        exercise = get_exercise(session_state.active_exercise_id)
        norm_lang = (language or "rw").strip().lower()
        if norm_lang not in SUPPORTED_EXERCISE_LANGUAGES:
            norm_lang = "rw"

        next_step_index = session_state.current_step_index + 1

        # Check if exercise is completed
        if next_step_index >= exercise.total_steps:
            session_state.complete_exercise()
            completion_question = exercise.get_completion_question(norm_lang)
            return LifecycleResult(
                response_text=completion_question,
                status=ExerciseStatus.COMPLETED,
                exercise_id=exercise.id,
                total_steps=exercise.total_steps,
                is_completed=True,
            )

        # Advance step
        session_state.advance_step(max_steps=exercise.total_steps)
        step = exercise.get_step(session_state.current_step_index)
        instruction = step.get_instruction(norm_lang)

        step_num = session_state.current_step_index + 1
        step_header = f"Step {step_num} of {exercise.total_steps}: "
        if norm_lang == "rw":
            step_header = f"Intambwe ya {step_num} kuri {exercise.total_steps}: "
        elif norm_lang == "fr":
            step_header = f"Étape {step_num} sur {exercise.total_steps} : "

        response_text = f"{step_header}{instruction}"
        return LifecycleResult(
            response_text=response_text,
            status=ExerciseStatus.IN_PROGRESS,
            exercise_id=exercise.id,
            step_number=step_num,
            total_steps=exercise.total_steps,
        )

    def stop_exercise(
        self,
        session_state: ExerciseSessionState,
        language: str,
    ) -> LifecycleResult:
        """
        Halts an active exercise upon user request and returns a calm check-in.
        """
        norm_lang = (language or "rw").strip().lower()
        if norm_lang not in SUPPORTED_EXERCISE_LANGUAGES:
            norm_lang = "rw"

        active_id = session_state.active_exercise_id
        session_state.abandon_exercise()

        ack_text = STOP_ACKNOWLEDGMENTS[norm_lang]
        return LifecycleResult(
            response_text=ack_text,
            status=ExerciseStatus.ABANDONED,
            exercise_id=active_id,
            is_stopped=True,
        )

    def decline_offer(
        self,
        session_state: ExerciseSessionState,
        language: str,
    ) -> LifecycleResult:
        """
        Processes user refusal of an offered exercise and applies the cooldown.
        """
        norm_lang = (language or "rw").strip().lower()
        if norm_lang not in SUPPORTED_EXERCISE_LANGUAGES:
            norm_lang = "rw"

        offered_id = session_state.offered_exercise_id
        session_state.decline_offer()

        ack_text = DECLINE_ACKNOWLEDGMENTS[norm_lang]
        return LifecycleResult(
            response_text=ack_text,
            status=ExerciseStatus.DECLINED,
            exercise_id=offered_id,
        )
