"""
MWIZERWA AI — Exercise Recommendation Subsystem
Provides deterministic, offline detection of explicit exercise requests and
rule-based evaluation of contextual exercise recommendations across Kinyarwanda,
English, and French.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.conversation.session import EligibleExerciseCandidate
from src.exercises.categories import ExerciseCategory
from src.exercises.recommendation_policy import unsolicited_offer_allowed
from src.exercises.suitability_selection import select_exercise_candidate
from src.exercises.models import ExerciseSessionState, SUPPORTED_EXERCISE_LANGUAGES
from src.llm.response_parser import InteractionIntent
from src.safety.categories import ApplicationAction, SafetyCategory


class RecommendationAction(str, Enum):
    """
    Decoupled action outcome produced by the exercise recommender.
    """
    NO_ACTION = "no_action"
    OFFER_EXERCISE = "offer_exercise"
    INITIATE_EXPLICIT_REQUEST = "initiate_explicit_request"
    CLARIFY_LANGUAGE = "clarify_language"
    CLARIFY_REQUEST = "clarify_request"


@dataclass(frozen=True)
class ExplicitRequestResult:
    """
    Result of evaluating user input for explicit exercise requests.
    """
    is_explicit_request: bool
    category: Optional[ExerciseCategory] = None
    matched_pattern: Optional[str] = None
    is_general_exercise_request: bool = False


@dataclass(frozen=True)
class RecommendationDecision:
    """
    Structured recommendation outcome evaluated by the recommender.
    """
    action: RecommendationAction
    exercise_id: Optional[str] = None
    category: Optional[ExerciseCategory] = None
    reason: str = ""


# Default mapping from category to canonical exercise id in registry
CATEGORY_DEFAULT_EXERCISE: Dict[ExerciseCategory, str] = {
    ExerciseCategory.BREATHING: "breathing_gentle_rhythm",
    ExerciseCategory.GROUNDING: "grounding_54321",
    ExerciseCategory.THOUGHT_REFLECTION: "reflection_gentle_perspective",
    ExerciseCategory.SELF_COMPASSION: "compassion_pause",
    ExerciseCategory.GRATITUDE_JOURNALING: "gratitude_grounded_anchor",
}


class ExerciseRecommender:
    """
    Evaluates whether an incoming user turn explicitly requests an exercise,
    or whether conversational context justifies offering an optional exercise.
    Operates offline without clinical scores or external dependencies.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compiles regex patterns for explicit requests and contextual cues."""

        # Explicit negation / refusal patterns
        # e.g., "I don't want an exercise", "Sinshaka umwitozo", "Pas d'exercice"
        self._negation_pattern = re.compile(
            r"\b((?:don'?t|do not|no|never|stop|not) (?:want|need|do|like|give me) (?:an? )?exercise|"
            r"sinshaka (?:umwitozo|imyitozo)|ntukore umwitozo|reka umwitozo|"
            r"(?:ne veux pas|pas d')\s*(?:d'exercice|d'exercices|faire un exercice))\b",
            re.IGNORECASE,
        )

        # Informational inquiry patterns (asking to explain or define, NOT requesting to perform)
        # e.g., "What is a breathing exercise?", "Ni iki umwitozo wo guhumeka?", "C'est quoi un exercice de respiration?"
        self._info_inquiry_pattern = re.compile(
            r"\b((?:what is|how does|can you explain|tell me about|define) (?:an? )?(?:breathing|grounding|wellness)? (?:exercise|technique)|"
            r"(?:ni iki|bisobanura iki|mbwira iby') (?:umwitozo|imyitozo)|"
            r"(?:qu'est-ce que|c'est quoi|explique-moi) (?:un |l')?(?:exercice|technique|ancrage|respiration))\b",
            re.IGNORECASE,
        )

        # Explicit Request Patterns per Category

        # 1. BREATHING:
        # EN: "can we do a breathing exercise", "help me breathe", "guide me through breathing"
        # RW: "dufatanye umwitozo wo guhumeka", "nfasha guhumeka", "kora umwitozo wo guhumeka"
        # FR: "faisons un exercice de respiration", "aide-moi à respirer", "guide-moi pour respirer"
        self._explicit_breathing = re.compile(
            r"\b((?:can we|let'?s|could we|i want to|please|help me) (?:do|try|practice) (?:a |the )?(?:deep )?breathing (?:exercise|technique)|"
            r"(?:help me|guide me (?:through|in)) (?:to )?breath(?:e|ing)|"
            r"(?:dufatanye|reka dukore|mfasha|nkore) umwitozo wo guhumeka|"
            r"mfasha guhumeka neza|"
            r"(?:faisons|pouvons-nous faire|je veux faire|aide-moi à faire) un exercice de respiration|"
            r"(?:aide-moi à|guide-moi pour) respirer)\b",
            re.IGNORECASE,
        )

        # 2. GROUNDING:
        # EN: "let's do grounding", "guide me through 5-4-3-2-1", "grounding exercise"
        # RW: "umwitozo wo gushyira ibitekerezo ku bikugose", "kora umwitozo wa 5-4-3-2-1"
        # FR: "faisons un exercice d'ancrage", "technique 5-4-3-2-1"
        self._explicit_grounding = re.compile(
            r"\b((?:can we|let'?s|could we|i want to|please|help me) (?:do|try|practice) (?:a |the )?(?:sensory |5-?4-?3-?2-?1 )?grounding(?: (?:exercise|technique))?|"
            r"(?:help me|guide me through) (?:sensory )?(?:grounding|5-?4-?3-?2-?1(?: technique)?)|"
            r"(?:dufatanye|reka dukore|mfasha) umwitozo wo gushyira ibitekerezo ku bikugose|"
            r"umwitozo wa 5-?4-?3-?2-?1|"
            r"(?:faisons|je veux faire|aide-moi à faire) un exercice d'ancrage|"
            r"(?:faisons|pratiquons|essayons) (?:la )?technique 5-?4-?3-?2-?1|"
            r"(?:technique|exercice) (?:de |d')?ancrage sensoriel)\b",
            re.IGNORECASE,
        )

        # 3. THOUGHT REFLECTION:
        # EN: "help me reframe my thoughts", "thought reflection exercise"
        # RW: "umwitozo wo gusuzuma ibitekerezo", "kwisubizaho amaso"
        # FR: "exercice de réflexion sur mes pensées", "m'aider à recadrer mes pensées"
        self._explicit_reflection = re.compile(
            r"\b((?:can we|let'?s|could we|i want to|help me) (?:do|try) (?:a )?(?:thought |cognitive )?reflection (?:exercise)?|"
            r"(?:help me|guide me to) (?:reframe|examine) (?:my )?thoughts|"
            r"(?:dufatanye|mfasha gukora) umwitozo wo gusuzuma ibitekerezo|"
            r"(?:faisons|aide-moi à faire) un exercice de réflexion sur (?:mes|les) pensées)\b",
            re.IGNORECASE,
        )

        # 4. SELF-COMPASSION:
        # EN: "self-compassion exercise", "help me be kinder to myself"
        # RW: "umwitozo wo kwiyitaho no kwiyumva", "kwigirira impuhwe"
        # FR: "exercice d'auto-compassion", "pause d'auto-compassion"
        self._explicit_compassion = re.compile(
            r"\b((?:can we|let'?s|could we|i want to|help me) (?:do|try) (?:a )?self-?compassion (?:exercise|pause)|"
            r"(?:help me|guide me to) be (?:more )?compassionate (?:to|with) myself|"
            r"(?:dufatanye|mfasha gukora) umwitozo wo kwiyitaho|"
            r"(?:faisons|aide-moi à faire) une pause d'auto-compassion|"
            r"exercice d'auto-compassion)\b",
            re.IGNORECASE,
        )

        # 5. GRATITUDE / ANCHOR:
        # EN: "gratitude exercise", "grounded anchor exercise", "gratitude reflection"
        # RW: "umwitozo wo kuzirikana ibyiza", "kuzirikana akantu ko kwiringira"
        # FR: "exercice de gratitude", "exercice de réconfort"
        self._explicit_gratitude = re.compile(
            r"\b((?:can we|let'?s|could we|i want to|help me) (?:do|try) (?:a )?(?:gratitude|grounded anchor) (?:exercise|reflection)|"
            r"(?:dufatanye|mfasha gukora) umwitozo wo kuzirikana (?:ibyiza|akantu)|"
            r"(?:faisons|aide-moi à faire) un exercice de gratitude)\b",
            re.IGNORECASE,
        )

        # 6. GENERAL EXERCISE REQUEST (No specific category mentioned):
        # e.g., "Can we do an exercise?", "Guide me through an exercise", "Dukore umwitozo", "Faisons un exercice"
        self._explicit_general = re.compile(
            r"\b((?:can we|let'?s|could we|i want to|please|help me) (?:do|try|start|give me) (?:a |an )?(?:calming |relaxation |wellness )?exercise|"
            r"(?:guide me through|lead me in) (?:a |an )?(?:calming |relaxation |wellness )?exercise|"
            r"(?:dufatanye|reka dukore|mfasha gukora) umwitozo|"
            r"(?:faisons|je veux faire|aide-moi à faire) un exercice)\b",
            re.IGNORECASE,
        )

        # Contextual Cues for Gentle Unsolicited Offers (Used strictly when intent is REFLECTION)
        # Sensation of physical tension / shallow breath -> Breathing
        self._cue_somatic_tension = re.compile(
            r"\b((?:i am|i feel|feeling) (?:so |very )?(?:tense|tight|wound up|physically stressed)|"
            r"(?:ndumva )?(?:ndembeje|umubiri uraremereye|umutima uri gutera cyane)|"
            r"(?:je me sens|je suis) (?:très |tellement )?(?:tendu|stressé physiquement|noué))\b",
            re.IGNORECASE,
        )

        # Sensation of racing thoughts / mental flooding -> Grounding
        self._cue_overwhelmed = re.compile(
            r"\b((?:my mind is racing|thoughts are spinning|feel completely overwhelmed|everything is spinning)|"
            r"(?:ibitekerezo biranyirukankana|umutwe uraremereye cyane|ndaremerewe n'ibitekerezo)|"
            r"(?:mes pensées se bousculent|je me sens submergé|la tête qui tourne))\b",
            re.IGNORECASE,
        )

        # Harsh self-criticism / guilt / self-blame -> Self-Compassion
        self._cue_self_blame = re.compile(
            r"\b((?:i am so hard on myself|i hate myself for this|i always fail|i feel like a failure)|"
            r"(?:ndinenga cyane|ndicira urubanza|ndi umunyantege nke cyane)|"
            r"(?:je m'en veux tellement|je suis trop dur avec moi-même|je me déteste pour ça))\b",
            re.IGNORECASE,
        )

    def detect_explicit_request(self, user_text: str) -> ExplicitRequestResult:
        """
        Determines whether user text contains a clear, explicit request to perform an exercise.

        Args:
            user_text: Raw incoming user utterance.

        Returns:
            ExplicitRequestResult indicating whether an explicit request was found and its category.
        """
        cleaned = (user_text or "").strip()
        if not cleaned:
            return ExplicitRequestResult(is_explicit_request=False)

        # Check for explicit negation / refusal
        if self._negation_pattern.search(cleaned):
            return ExplicitRequestResult(is_explicit_request=False)

        # Check for informational inquiries (e.g. "What is a breathing exercise?")
        if self._info_inquiry_pattern.search(cleaned):
            return ExplicitRequestResult(is_explicit_request=False)

        # Category-specific explicit requests
        if self._explicit_breathing.search(cleaned):
            return ExplicitRequestResult(
                is_explicit_request=True,
                category=ExerciseCategory.BREATHING,
                matched_pattern="breathing",
            )

        if self._explicit_grounding.search(cleaned):
            return ExplicitRequestResult(
                is_explicit_request=True,
                category=ExerciseCategory.GROUNDING,
                matched_pattern="grounding",
            )

        if self._explicit_reflection.search(cleaned):
            return ExplicitRequestResult(
                is_explicit_request=True,
                category=ExerciseCategory.THOUGHT_REFLECTION,
                matched_pattern="thought_reflection",
            )

        if self._explicit_compassion.search(cleaned):
            return ExplicitRequestResult(
                is_explicit_request=True,
                category=ExerciseCategory.SELF_COMPASSION,
                matched_pattern="self_compassion",
            )

        if self._explicit_gratitude.search(cleaned):
            return ExplicitRequestResult(
                is_explicit_request=True,
                category=ExerciseCategory.GRATITUDE_JOURNALING,
                matched_pattern="gratitude_journaling",
            )

        # General un-categorized exercise request (e.g. "Can we do an exercise?")
        if self._explicit_general.search(cleaned):
            return ExplicitRequestResult(
                is_explicit_request=True,
                category=None,
                matched_pattern="general_exercise",
                is_general_exercise_request=True,
            )

        return ExplicitRequestResult(is_explicit_request=False)

    def default_standalone_candidates(self) -> List["EligibleExerciseCandidate"]:
        from src.conversation.session import EligibleExerciseCandidate

        return [
            EligibleExerciseCandidate(id=ex_id, category=category)
            for category, ex_id in CATEGORY_DEFAULT_EXERCISE.items()
        ]

    def exercise_reluctance_detected(
        self,
        user_text: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        texts = [user_text]
        for turn in (conversation_context or {}).get("recent_turns") or []:
            if turn.get("role") == "user":
                content = (turn.get("content") or "").strip()
                if content:
                    texts.append(content)
        for text in texts:
            if self._negation_pattern.search(text):
                return True
        return False

    def _pick_exercise_id(
        self,
        category: ExerciseCategory,
        eligible_exercises: Optional[List["EligibleExerciseCandidate"]],
    ) -> Optional[str]:
        if eligible_exercises is not None:
            for candidate in eligible_exercises:
                if candidate.category == category:
                    return candidate.id
            return None
        return CATEGORY_DEFAULT_EXERCISE.get(category)

    def evaluate_recommendation(
        self,
        user_text: str,
        interaction_intent: Optional[InteractionIntent],
        session_state: ExerciseSessionState,
        language: Optional[str] = None,
        eligible_exercises: Optional[List["EligibleExerciseCandidate"]] = None,
        *,
        llm=None,
        conversation_context: Optional[Dict[str, Any]] = None,
        safety_action: ApplicationAction = ApplicationAction.NORMAL_CONVERSATION,
        safety_category: SafetyCategory = SafetyCategory.SAFE_CONVERSATIONAL,
    ) -> RecommendationDecision:
        """Unsolicited exercise offer evaluation (explicit requests handled in ConversationEngine)."""
        cleaned = (user_text or "").strip()
        if not cleaned:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="Empty user input.",
            )

        if not unsolicited_offer_allowed(safety_action, safety_category):
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="Gate 1 policy suppresses unsolicited exercise offers.",
            )

        backend_candidates = eligible_exercises is not None
        if backend_candidates and not eligible_exercises:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="No backend-supplied eligible exercises.",
            )

        # LISTENING: Never make unsolicited exercise offers
        if interaction_intent == InteractionIntent.LISTENING:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="Unsolicited offers suppressed during active listening (LISTENING intent).",
            )

        # PRACTICAL_ADVICE: General practical advice does not qualify for exercise offers
        if interaction_intent == InteractionIntent.PRACTICAL_ADVICE:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="General practical advice request does not qualify for unsolicited exercise offer.",
            )

        # UNCLEAR: Clarify instead of offering an exercise
        if interaction_intent == InteractionIntent.UNCLEAR:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="Conversational intent is UNCLEAR; unsolicited offers suppressed.",
            )
        if interaction_intent != InteractionIntent.REFLECTION:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="Unsolicited offers require REFLECTION interaction intent.",
            )

        if self.exercise_reluctance_detected(cleaned, conversation_context):
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason="User reluctance toward exercises detected in recent context.",
            )

        if not backend_candidates and session_state.cooldown_turns_remaining > 0:
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason=f"Unsolicited offer suppressed: cooldown active ({session_state.cooldown_turns_remaining} turns remaining).",
            )

        candidates = (
            list(eligible_exercises)
            if eligible_exercises is not None
            else self.default_standalone_candidates()
        )

        selected_id = select_exercise_candidate(
            llm,
            user_text=cleaned,
            interaction_intent=interaction_intent,
            language=language,
            candidates=candidates,
            conversation_context=conversation_context or {},
        )
        if not selected_id:
            if llm is None or not hasattr(llm, "generate_auxiliary_json"):
                reason = "Suitability inference unavailable for unsolicited offers."
            else:
                reason = "Suitability model selected no appropriate exercise."
            return RecommendationDecision(
                action=RecommendationAction.NO_ACTION,
                reason=reason,
            )
        category = next((c.category for c in candidates if c.id == selected_id), None)
        return RecommendationDecision(
            action=RecommendationAction.OFFER_EXERCISE,
            exercise_id=selected_id,
            category=category,
            reason="Structured suitability selection chose an eligible exercise.",
        )
