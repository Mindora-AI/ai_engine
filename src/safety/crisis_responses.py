"""
MWIZERWA AI — Crisis Response Templates
Deterministic, verified, speakable responses for safety overrides across Kinyarwanda, English, and French.
"""

from enum import Enum
from typing import Dict, Tuple


class ResponseSituation(str, Enum):
    """
    Situational contexts requiring specialized or deterministic crisis intervention responses.
    """
    PASSIVE_DEATH_WISH = "passive_death_wish"
    SELF_HARM_THOUGHTS = "self_harm_thoughts"
    THIRD_PERSON_CRISIS = "third_person_crisis"
    IMMINENT_SELF_HARM = "imminent_self_harm"
    IMMINENT_VIOLENCE_OTHERS = "imminent_violence_others"
    ONGOING_PHYSICAL_DANGER = "ongoing_physical_danger"
    ACUTE_MEDICAL_EMERGENCY = "acute_medical_emergency"


class TemplateNotFoundError(KeyError):
    """Raised when a crisis response template cannot be found for the specified situation or language."""
    pass


# Deterministic crisis response templates.
# Each entry is speakable, empathetic yet direct, free of markdown, and grounded in verified local emergency contacts.
_CRISIS_TEMPLATES: Dict[Tuple[ResponseSituation, str], str] = {
    # --- PASSIVE DEATH WISH ---
    (ResponseSituation.PASSIVE_DEATH_WISH, "rw"): (
        "Ndakumva kandi ndakwitayeho. Biragaragara ko unaniwe cyane mu mutima. "
        "Nturi wenyine muri ibi bibazo, kandi ubuzima bwawe bufite agaciro gakomeye. "
        "Mbwira mu bwisanzure: ese ubu waba uri gutekereza kwigirira nabi cyangwa kwiyahura? "
        "Nyamuneka gerageza kubiganiriza umuntu wiringira cyangwa umuganga ushobora kugufasha."
    ),
    (ResponseSituation.PASSIVE_DEATH_WISH, "en"): (
        "I hear how exhausted you feel, and I want you to know that your life matters deeply. "
        "You do not have to carry this heavy burden alone. "
        "Can you tell me directly: are you currently thinking about harming yourself or ending your life? "
        "Please reach out to someone you trust, a close friend, or a healthcare professional who can support you."
    ),
    (ResponseSituation.PASSIVE_DEATH_WISH, "fr"): (
        "Je comprends à quel point vous êtes épuisé et votre vie a une valeur précieuse. "
        "Vous ne devez pas porter ce lourd fardeau tout seul. "
        "Dites-moi en toute franchise : êtes-vous actuellement en train de penser à vous faire du mal ou à mettre fin à vos jours ? "
        "N'hésitez pas à vous confier à une personne de confiance ou à un professionnel de santé."
    ),

    # --- SELF HARM THOUGHTS ---
    (ResponseSituation.SELF_HARM_THOUGHTS, "rw"): (
        "Mbabajwe cyane n'uko wumva umeze gutyo, ariko ubuzima bwawe bufite agaciro kanini. "
        "Nyamuneka ntugire icyo wikoraho cyangwa ngo ube wenyine muri ibi. "
        "Ganira n'umuntu wiringira, inshuti, cyangwa umuganga ushobora kugufasha ubu."
    ),
    (ResponseSituation.SELF_HARM_THOUGHTS, "en"): (
        "I am deeply concerned about what you are going through, and your safety is the most important thing. "
        "Please do not hurt yourself. "
        "Reach out right away to someone you trust, a loved one, or a professional who can be there for you right now."
    ),
    (ResponseSituation.SELF_HARM_THOUGHTS, "fr"): (
        "Je suis profondément touché par ce que vous traversez et votre sécurité est primordiale. "
        "S'il vous plaît, ne vous faites pas de mal. "
        "Parlez immédiatement à un proche de confiance ou à un professionnel qui pourra vous soutenir."
    ),

    # --- THIRD PERSON CRISIS ---
    (ResponseSituation.THIRD_PERSON_CRISIS, "rw"): (
        "Birakomeye kubona umuntu ukunda ahangayitse muri ubu buryo. "
        "Nyamuneka komeza kumuba hafi kandi ntumusige wenyine. "
        "Niba ari mu kaga gakomeye, hamagara ubutabazi kuri nimero 112 cyangwa 912."
    ),
    (ResponseSituation.THIRD_PERSON_CRISIS, "en"): (
        "It is deeply worrying to see someone you care about in distress. "
        "Please do not leave them alone. "
        "Stay with them and contact local emergency services immediately at 112 or medical help at 912 if they are in danger."
    ),
    (ResponseSituation.THIRD_PERSON_CRISIS, "fr"): (
        "Il est très éprouvant de voir un proche en grande détresse. "
        "S'il vous plaît, ne le laissez pas seul. "
        "Restez auprès de cette personne et, si elle est en danger, contactez sans attendre les urgences au 112 ou l'aide médicale au 912."
    ),

    # --- IMMINENT SELF HARM ---
    (ResponseSituation.IMMINENT_SELF_HARM, "rw"): (
        "Ubu uri mu kaga kandi ubuzima bwawe bufite agaciro gakomeye cyane. "
        "Nyamuneka tabaza byihuse. "
        "Hamagara nimero y'ubutabazi ya Polisi 112 cyangwa ubutabazi bw'ubuvuzi bwa SAMU kuri 912 ako kanya, "
        "cyangwa ubwire umuntu uri hafi yawe agutabare."
    ),
    (ResponseSituation.IMMINENT_SELF_HARM, "en"): (
        "You are in immediate distress and your life is profoundly valuable. "
        "Please seek help immediately. "
        "Call emergency services at 112 or SAMU ambulance services at 912 right now, "
        "or reach out to someone nearby who can stay with you."
    ),
    (ResponseSituation.IMMINENT_SELF_HARM, "fr"): (
        "Vous êtes en danger immédiat et votre vie compte énormément. "
        "Demandez de l'aide sans attendre. "
        "Appelez immédiatement les urgences au 112 ou le SAMU au 912, "
        "ou alertez une personne proche à vos côtés."
    ),

    # --- IMMINENT VIOLENCE OTHERS ---
    (ResponseSituation.IMMINENT_VIOLENCE_OTHERS, "rw"): (
        "Umutekano n'ubuzima by'abantu birihutirwa cyane. "
        "Niba hari umuntu uri mu kaga k'urugomo cyangwa ibikorwa by'ubugizi bwa nabi, "
        "hamagara Polisi y'u Rwanda byihuse kuri nimero 112 kugira ngo batange ubutabazi."
    ),
    (ResponseSituation.IMMINENT_VIOLENCE_OTHERS, "en"): (
        "Safety and human life must be protected immediately. "
        "If anyone is in immediate physical danger, "
        "call the emergency line at 112 right now to request urgent assistance."
    ),
    (ResponseSituation.IMMINENT_VIOLENCE_OTHERS, "fr"): (
        "La sécurité et la vie humaine doivent être protégées d'urgence. "
        "Si quelqu'un est en danger physique immédiat, "
        "veuillez appeler sans attendre les urgences au 112 pour une intervention."
    ),

    # --- ONGOING PHYSICAL DANGER ---
    (ResponseSituation.ONGOING_PHYSICAL_DANGER, "rw"): (
        "Niba uri ahantu hari akaga k'umutekano cyangwa inkongi y'umuriro, vaho vuba ujye ahantu hatekanye. "
        "Hamagara ubutabazi bwihuse bwa Polisi kuri nimero 112 ako kanya ubone ubutabazi."
    ),
    (ResponseSituation.ONGOING_PHYSICAL_DANGER, "en"): (
        "If you are in immediate physical danger, please move to a safe place right away. "
        "Call the Rwanda National Police emergency hotline at 112 immediately for urgent help."
    ),
    (ResponseSituation.ONGOING_PHYSICAL_DANGER, "fr"): (
        "Si vous êtes en danger physique immédiat, mettez-vous vite en lieu sûr. "
        "Contactez immédiatement la Police Nationale au 112 pour obtenir une aide d'urgence."
    ),

    # --- ACUTE MEDICAL EMERGENCY ---
    (ResponseSituation.ACUTE_MEDICAL_EMERGENCY, "rw"): (
        "Ibi birasaba ubutabazi bwihuse bw'ubuvuzi. "
        "Nyamuneka hamagara imbangukiragutabara ya SAMU kuri nimero 912 cyangwa Polisi kuri 112 ako kanya "
        "kugira ngo uhabwe ubufasha bw'ubuvuzi bwihuse."
    ),
    (ResponseSituation.ACUTE_MEDICAL_EMERGENCY, "en"): (
        "This requires urgent medical care. "
        "Please call SAMU ambulance services immediately at 912 or emergency services at 112 right now "
        "to get medical attention."
    ),
    (ResponseSituation.ACUTE_MEDICAL_EMERGENCY, "fr"): (
        "Cette situation exige une assistance médicale urgente. "
        "Appelez sans attendre le SAMU au 912 ou les services d'urgence au 112 "
        "pour recevoir des soins médicaux immédiats."
    ),
}

SUPPORTED_SAFETY_LANGUAGES: Tuple[str, ...] = ("rw", "en", "fr")


def get_safety_response(situation: ResponseSituation, language: str) -> str:
    """
    Retrieve a deterministic safety response for a situation and target language.

    Args:
        situation: ResponseSituation enum member.
        language: Language code ('rw', 'en', 'fr').

    Returns:
        Deterministic response string.

    Raises:
        ValueError: If language is not one of ('rw', 'en', 'fr').
        TemplateNotFoundError: If template is missing for the combination.
    """
    if not isinstance(situation, ResponseSituation):
        raise ValueError(f"Invalid situation: {situation}. Expected ResponseSituation enum.")

    norm_lang = (language or "").strip().lower()
    if norm_lang not in SUPPORTED_SAFETY_LANGUAGES:
        raise ValueError(
            f"Unsupported safety template language: '{language}'. "
            f"Supported languages are: {SUPPORTED_SAFETY_LANGUAGES}."
        )

    key = (situation, norm_lang)
    if key not in _CRISIS_TEMPLATES:
        raise TemplateNotFoundError(
            f"No crisis response template found for situation '{situation.value}' and language '{norm_lang}'."
        )

    return _CRISIS_TEMPLATES[key]


def get_available_situations() -> Tuple[ResponseSituation, ...]:
    """Return all defined response situations."""
    return tuple(ResponseSituation)


def get_supported_languages() -> Tuple[str, ...]:
    """Return all supported language codes for crisis templates."""
    return SUPPORTED_SAFETY_LANGUAGES
