"""
MWIZERWA AI — Static Exercise Registry
Provides the single, authoritative, offline catalogue of verified wellness exercises
across Kinyarwanda, English, and French.
"""

from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from src.exercises.categories import ExerciseCategory
from src.exercises.models import Exercise, ExerciseStep


class ExerciseNotFoundError(KeyError):
    """Raised when an exercise identifier cannot be found in the registry."""
    pass


# ---------------------------------------------------------------------------
# Static Catalogue of the Five Initial Exercises
# ---------------------------------------------------------------------------

def _register_initial_exercises() -> Dict[str, Exercise]:
    catalog: Dict[str, Exercise] = {}

    # 1. BREATHING: Gentle Rhythmic Breathing
    breathing = Exercise(
        id="breathing_gentle_rhythm",
        category=ExerciseCategory.BREATHING,
        name_rw="Guhumeka mu ituze no mu bwitonzi",
        name_en="Gentle Rhythmic Breathing",
        name_fr="Respiration douce et rythmée",
        description_rw="Umwitozo woroheje wo guhumeka buhoro buhoro ufasha umubiri n'ubwenge gucururuka no gutuza.",
        description_en="A gentle breathing exercise that helps ease physical tension and brings a sense of calm.",
        description_fr="Un exercice simple de respiration lente pour apaiser les tensions physiques et retrouver le calme.",
        estimated_duration_minutes=2,
        cautionary_notes=(
            "Niba wumva uzungerezwa cyangwa uhangayitse, hagarika maze usubire mu guhumeka bisanzwe.",
            "Stop immediately if you feel dizzy or lightheaded; return to your natural, comfortable pace.",
            "Cessez l'exercice en cas de vertiges ou d'inconfort ; reprenez votre respiration naturelle.",
        ),
        completion_question_rw="Wumvise bimeze bite mu mubiri wawe nyuma yo guhumeka gutyo?",
        completion_question_en="How does your body feel after taking those gentle breaths?",
        completion_question_fr="Comment vous sentez-vous dans votre corps après ces quelques respirations douces ?",
        steps=(
            ExerciseStep(
                step_number=1,
                instruction_rw=(
                    "Fata umwanya muto wicare neza cyangwa uhagarare mu buryo bukworoheye. "
                    "Niba wumva bibangamye cyangwa uzungerezwa igihe icyo aricyo cyose, "
                    "hagarika uyu mwitozo maze usubire mu guhumeka kwawe bisanzwe."
                ),
                instruction_en=(
                    "Find a comfortable position, sitting or standing naturally. "
                    "If you feel uncomfortable, dizzy, or lightheaded at any point, "
                    "feel free to stop immediately and return to your natural breathing."
                ),
                instruction_fr=(
                    "Installez-vous confortablement, assis ou debout. "
                    "Si vous ressentez une gêne ou un vertige à tout moment, "
                    "arrêtez immédiatement et reprenez votre respiration naturelle."
                ),
                duration_seconds=15,
            ),
            ExerciseStep(
                step_number=2,
                instruction_rw=(
                    "Humeka buhoro buhoro winjiza umwuka mu mazuru mu buryo butuje. "
                    "Fata umwuka gato mu gihe gito cyane kitakubangamiye, nta gahato."
                ),
                instruction_en=(
                    "Slowly breathe in through your nose at an easy, unhurried pace. "
                    "Pause for just a brief, comfortable moment without forcing or holding your breath tightly."
                ),
                instruction_fr=(
                    "Inspirez lentement par le nez, à votre propre rythme. "
                    "Marquez une très brève pause confortable, sans forcer ni bloquer l'air."
                ),
                duration_seconds=20,
            ),
            ExerciseStep(
                step_number=3,
                instruction_rw=(
                    "Sohobora umwuka buhoro mu kanwa urekure amaganya yose. "
                    "Reka umubiri wawe ugende wumva utuje buhoro buhoro."
                ),
                instruction_en=(
                    "Gently release the breath out through your mouth, "
                    "letting your shoulders drop and relaxing your muscles as you exhale."
                ),
                instruction_fr=(
                    "Expirez doucement par la bouche, "
                    "en laissant vos épaules s'abaisser et votre corps se détendre paisiblement."
                ),
                duration_seconds=20,
            ),
            ExerciseStep(
                step_number=4,
                instruction_rw=(
                    "Ongera ubikore indi nshuro imwe cyangwa ebyiri mu ituze ryawe bwite, "
                    "maze ureke guhumeka gusubire mu buryo busanzwe."
                ),
                instruction_en=(
                    "Take one or two more gentle breaths at your own natural pace, "
                    "allowing your body to settle back into its ordinary, comfortable breathing."
                ),
                instruction_fr=(
                    "Prenez encore une ou deux respirations douces à votre rythme, "
                    "puis laissez votre corps retrouver son souffle habituel et naturel."
                ),
                duration_seconds=25,
            ),
        ),
    )
    catalog[breathing.id] = breathing

    # 2. GROUNDING: 5-4-3-2-1 Sensory Grounding
    grounding = Exercise(
        id="grounding_54321",
        category=ExerciseCategory.GROUNDING,
        name_rw="Gushyira ibitekerezo ku bikugose (5-4-3-2-1)",
        name_en="5-4-3-2-1 Sensory Grounding",
        name_fr="Ancrage sensoriel 5-4-3-2-1",
        description_rw="Umwitozo wo gushyira ibitekerezo ku bintu bikugose hifashishijwe ingingo z'umubiri, ugufasha kutiruka mu bitekerezo.",
        description_en="A gentle grounding exercise using your senses to bring your awareness back to the present surroundings.",
        description_fr="Un exercice d'ancrage sensoriel pour ramener l'attention au moment présent et calmer le flux des pensées.",
        estimated_duration_minutes=3,
        cautionary_notes=(
            "Ushobora gusimbuka cyangwa guhindura ingingo udashoboye gukoresha neza, nta kibazo.",
            "You may adapt or skip any senses you cannot comfortably use; no physical movement is required.",
            "Vous pouvez adapter ou ignorer tout sens inconfortable ; aucun mouvement physique n'est requis.",
        ),
        completion_question_rw="Ese wumvise byagufashije bite gushyira ibitekerezo ku bikugose?",
        completion_question_en="How did that exercise feel for your focus and awareness?",
        completion_question_fr="Comment avez-vous vécu ce moment d'attention à votre environnement ?",
        steps=(
            ExerciseStep(
                step_number=1,
                instruction_rw=(
                    "Reba ibikugose utuje maze uvuge ibintu bitanu ushobora kubona muri aka kanya. "
                    "Niba utareba neza cyangwa bihagoranye, ushobora kwibuka ibintu bitanu uzi neza cyangwa ugakomeza ku zindi ngingo."
                ),
                instruction_en=(
                    "Take a calm look around your space and notice five things you can see right now. "
                    "If visual observation is difficult, feel free to imagine five familiar objects or skip to senses that feel comfortable."
                ),
                instruction_fr=(
                    "Regardez calmement autour de vous et repérez cinq choses que vous pouvez voir. "
                    "Si cela est difficile, vous pouvez imaginer cinq objets familiers ou passer à un autre sens qui vous convient."
                ),
                duration_seconds=30,
            ),
            ExerciseStep(
                step_number=2,
                instruction_rw=(
                    "Noneho itegereze ibintu bine ushobora gukoraho cyangwa kumva ku mubiri wawe "
                    "(urugero: imyenda wambaye, intebe wicayemo, cyangwa hasi munsi y'ibirenge)."
                ),
                instruction_en=(
                    "Next, notice four things you can physically feel or touch "
                    "(such as the texture of your clothes, the chair supporting you, or your feet resting on the ground)."
                ),
                instruction_fr=(
                    "À présent, remarquez quatre sensations physiques au contact de votre corps "
                    "(la texture d'un vêtement, le soutien de votre siège, ou vos pieds posés au sol)."
                ),
                duration_seconds=30,
            ),
            ExerciseStep(
                step_number=3,
                instruction_rw=(
                    "Tega amatwi utuje maze wumve ibintu bitatu wumva muri aka kanya "
                    "(urugero: amajwi y'ibinyabiziga, akayaga, cyangwa umwuka uhita)."
                ),
                instruction_en=(
                    "Gently listen to your environment and identify three sounds you can hear right now, near or far."
                ),
                instruction_fr=(
                    "Écoutez attentivement votre environnement et identifiez trois sons que vous percevez, qu'ils soient proches ou lointains."
                ),
                duration_seconds=30,
            ),
            ExerciseStep(
                step_number=4,
                instruction_rw=(
                    "Itegereze ibintu bibiri waba uri kumva impumuro yabyo, "
                    "cyangwa impumuro ebyiri zikunze kuguha ituze igihe uzitekereje."
                ),
                instruction_en=(
                    "Notice two scents you can smell around you, "
                    "or recall two comforting scents that bring you a sense of ease."
                ),
                instruction_fr=(
                    "Remarquez deux odeurs autour de vous, "
                    "ou pensez à deux parfums familiers qui vous apportent du réconfort."
                ),
                duration_seconds=25,
            ),
            ExerciseStep(
                step_number=5,
                instruction_rw=(
                    "Gusoza, fata umwuka umwe utuje cyangwa utegereze uburyohe bumwe wumva mu kanwa, "
                    "maze uzirikane ko muri aka kanya uri ahantu hatekanye."
                ),
                instruction_en=(
                    "Finally, notice one taste in your mouth, or take one slow, grounding breath, acknowledging this present moment."
                ),
                instruction_fr=(
                    "Enfin, prenez conscience d'un goût dans votre bouche ou prenez une inspiration douce, en revenant simplement au moment présent."
                ),
                duration_seconds=25,
            ),
        ),
    )
    catalog[grounding.id] = grounding

    # 3. THOUGHT_REFLECTION: Gentle Thought Reflection
    reflection = Exercise(
        id="reflection_gentle_perspective",
        category=ExerciseCategory.THOUGHT_REFLECTION,
        name_rw="Kwisubizaho amaso mu bwitonzi",
        name_en="Gentle Thought Reflection",
        name_fr="Réflexion douce sur les pensées",
        description_rw="Uburyo butuje bwo gusuzuma ibitekerezo bikuremereye utiriwe ubyicira urubanza cyangwa ngo ubihangane.",
        description_en="A calm, non-judgmental pause to explore heavy thoughts with curiosity and kindness.",
        description_fr="Une pause bienveillante pour observer les pensées pesantes sans jugement et avec douceur.",
        estimated_duration_minutes=3,
        cautionary_notes=(
            "Ntabwo uyu mwitozo ugamije kuvuguruza ibyo utekereza, ahubwo ni ukureba mu bwitonzi.",
            "This exercise does not judge your thoughts or claim they are wrong; it offers a gentle moment to look again.",
            "Cet exercice ne juge pas vos pensées ; il offre simplement un regard doux et bienveillant.",
        ),
        completion_question_rw="Bimeze bite kurebera icyo gitekerezo mu bundi buryo bworoheje?",
        completion_question_en="What was it like to look at that thought with a bit more gentleness?",
        completion_question_fr="Qu'avez-vous ressenti en observant cette pensée avec plus de douceur ?",
        steps=(
            ExerciseStep(
                step_number=1,
                instruction_rw=(
                    "Fata umwanya muto wibaze: ni ikihe gitekerezo cyangwa impungenge uri kumva iguremereye muri aka kanya? "
                    "Kirebe gusa mu mutima utiriwe ucyicira urubanza cyangwa ngo wumve ko utagombaga kugitekereza."
                ),
                instruction_en=(
                    "Take a quiet moment to bring to mind one thought or worry that is feeling heavy right now. "
                    "Simply observe it without judging yourself for having it."
                ),
                instruction_fr=(
                    "Prenez un instant pour observer une pensée ou une inquiétude qui vous pèse en ce moment. "
                    "Regardez-la simplement, sans vous juger de la ressentir."
                ),
                duration_seconds=30,
            ),
            ExerciseStep(
                step_number=2,
                instruction_rw=(
                    "Ibaze mu bwitonzi: ese haba hari indi nzira yoroheje cyangwa ituje yo kurebamo iki kintu, "
                    "nk'uko wabiganiriza inshuti magara?"
                ),
                instruction_en=(
                    "Ask yourself gently: is there another way to view this situation, "
                    "perhaps with the same fairness and understanding you would offer a trusted friend?"
                ),
                instruction_fr=(
                    "Demandez-vous avec bienveillance : existe-t-il une autre façon de regarder cette situation, "
                    "comme vous le feriez pour un ami cher ?"
                ),
                duration_seconds=35,
            ),
            ExerciseStep(
                step_number=3,
                instruction_rw=(
                    "Hitamo igitekerezo kimwe gito koroheje ushobora gukomerezaho ubu, "
                    "wibuka ko udakeneye gukemura ibibazo byose muri aka kanya."
                ),
                instruction_en=(
                    "Allow yourself to hold one small, balanced thought that offers a little more breathing room, "
                    "remembering that not everything needs to be resolved right now."
                ),
                instruction_fr=(
                    "Accordez-vous une pensée plus équilibrée et apaisante, "
                    "en vous rappelant que tout n'a pas besoin d'être résolu immédiatement."
                ),
                duration_seconds=35,
            ),
        ),
    )
    catalog[reflection.id] = reflection

    # 4. SELF_COMPASSION: Self-Compassion Pause
    compassion = Exercise(
        id="compassion_pause",
        category=ExerciseCategory.SELF_COMPASSION,
        name_rw="Gufata umwanya wo kwiyitaho no kwiyumva",
        name_en="Self-Compassion Pause",
        name_fr="Pause d'auto-compassion",
        description_rw="Guhagarara gato ukakirana ibyo urimo no kwiyitaho mu gihe wumva unaniwe cyangwa urushye.",
        description_en="A brief pause to acknowledge current struggles with warmth and understanding, without forcing positivity.",
        description_fr="Un court moment pour reconnaître ses difficultés avec bienveillance, sans imposer de positivité forcée.",
        estimated_duration_minutes=2,
        cautionary_notes=(
            "Ntugahatire umutima kumererwa neza ku ngufu; akira ibyo urimo uko biri muri aka kanya.",
            "Do not force yourself to feel cheerful; simply acknowledge that things are difficult right now.",
            "Ne forcez pas d'émotions positives ; reconnaissez simplement ce qui est difficile en ce moment.",
        ),
        completion_question_rw="Wumvise bimeze bite guha umwanya ibyo urimo utiriwe wiyereka ko byose bimeze neza?",
        completion_question_en="How did it feel to pause and acknowledge things just as they are right now?",
        completion_question_fr="Comment avez-vous vécu ce moment de bienveillance envers vous-même ?",
        steps=(
            ExerciseStep(
                step_number=1,
                instruction_rw=(
                    "Zirikana ko ibyo urimo ubu bikomeye cyangwa binanije. "
                    "Kora ikimenyetso cyo kwiyumva, nko kwambika ikiganza ku gituza cyangwa gufata intoki, wibwire uti: 'Ibi biragoye ubu.'"
                ),
                instruction_en=(
                    "Gently acknowledge that this moment is difficult or exhausting. "
                    "You might place a hand over your heart or on your arm, and softly say to yourself: 'This is a tough moment.'"
                ),
                instruction_fr=(
                    "Reconnaissez doucement que ce moment est difficile ou éprouvant. "
                    "Vous pouvez poser une main sur votre cœur ou votre bras, et vous dire : 'C'est un moment délicat.'"
                ),
                duration_seconds=25,
            ),
            ExerciseStep(
                step_number=2,
                instruction_rw=(
                    "Ibuke ko amakuba, gucika intege, no kumva unaniwe ari ibintu bisanzwe bibaho mu buzima bwa muntu. "
                    "Nturi wenyine muri ibi byiyumvo."
                ),
                instruction_en=(
                    "Remind yourself that struggle, exhaustion, and feeling imperfect are part of our shared human experience. "
                    "You are not alone in feeling this way."
                ),
                instruction_fr=(
                    "Rappelez-vous que la fatigue, le doute et les épreuves font partie de l'expérience humaine. "
                    "Vous n'êtes pas seul à traverser cela."
                ),
                duration_seconds=25,
            ),
            ExerciseStep(
                step_number=3,
                instruction_rw=(
                    "Muri aka kanya, wibwire amagambo yuje impuhwe wakagombye kubwira umuntu ukunda cyane, "
                    "utiriwe wisaba guhita wishima cyangwa ngo uhindure ibyo wumva."
                ),
                instruction_en=(
                    "Offer yourself one supportive sentence you would share with someone you care about, "
                    "without expecting yourself to suddenly feel happy or cheerful."
                ),
                instruction_fr=(
                    "Offrez-vous une parole de soutien que vous adresseriez à quelqu'un que vous aimez, "
                    "sans exiger de vous-même d'aller immédiatement mieux."
                ),
                duration_seconds=30,
            ),
        ),
    )
    catalog[compassion.id] = compassion

    # 5. GRATITUDE_JOURNALING: Grounded Anchor of Comfort
    gratitude = Exercise(
        id="gratitude_grounded_anchor",
        category=ExerciseCategory.GRATITUDE_JOURNALING,
        name_rw="Kuzirikana akantu gato ko kwiringira",
        name_en="Grounded Anchor of Comfort",
        name_fr="Point d'ancrage et de réconfort",
        description_rw="Kuzirikana ikintu kimwe gito kandi gifatika cyakuzaniye agahenge cyangwa ituze uyu munsi, utiriwe wibeshya.",
        description_en="Recognizing one real, concrete detail that offered a moment of relief, comfort, or safety today, without forced optimism.",
        description_fr="Identifier un détail concret et réel qui a apporté un instant de répit ou de sécurité aujourd'hui, sans optimisme forcé.",
        estimated_duration_minutes=2,
        cautionary_notes=(
            "Ntusabwa gushimira ibintu bibabaje cyangwa ngo wibwire ko byose ari byiza.",
            "You are never asked to feel grateful for painful circumstances or pretend that everything is fine.",
            "Il n'est jamais question d'éprouver de la gratitude pour des souffrances ni de prétendre que tout va bien.",
        ),
        completion_question_rw="Ese byagenze bite kuzirikana ako kantu gato k'agahenge?",
        completion_question_en="How was it to pause and notice that small anchor of comfort?",
        completion_question_fr="Qu'avez-vous ressenti en portant attention à ce petit moment de répit ?",
        steps=(
            ExerciseStep(
                step_number=1,
                instruction_rw=(
                    "Tekereza ku munsi wa none maze urebe niba hari akantu kamwe gato gafatika kakuzaniye ituze, "
                    "uburuhukiro, cyangwa umutekano (urugero: ikirahure cy'amazi akonje, ijwi ry'inshuti, akayaga, "
                    "cyangwa umwanya wo kuruhuka). Ntibisaba ko umunsi wose waba wagenze neza."
                ),
                instruction_en=(
                    "Think back through today and notice one small, concrete detail that offered even a fleeting moment "
                    "of relief, comfort, or safety (such as a warm drink, a supportive voice, or a moment of rest). "
                    "It does not require that the day was good or easy."
                ),
                instruction_fr=(
                    "Pensez à votre journée et repérez un détail simple et concret qui a offert un instant de répit, "
                    "de confort ou de sécurité (une boisson chaude, une voix familière, ou un temps de repos). "
                    "Cela ne suppose pas que la journée ait été facile."
                ),
                duration_seconds=30,
            ),
            ExerciseStep(
                step_number=2,
                instruction_rw=(
                    "Fata umwanya muto wumve ukuntu ako kantu gato koroheje kakugiriye neza, "
                    "wibuka ko ari uburenganzira bwawe kwakira umwanya w'agahenge no mu bihe bigoye."
                ),
                instruction_en=(
                    "Spend a quiet moment noticing what that small bit of comfort felt like, "
                    "honoring your right to experience moments of ease even during difficult times."
                ),
                instruction_fr=(
                    "Prenez un instant pour ressentir ce petit réconfort, "
                    "en accueillant ce moment d'apaisement sans rien forcer d'autre."
                ),
                duration_seconds=30,
            ),
        ),
    )
    catalog[gratitude.id] = gratitude

    return catalog


_EXERCISES: Mapping[str, Exercise] = MappingProxyType(_register_initial_exercises())


def is_registered_exercise_id(exercise_id: str) -> bool:
    """Return True when exercise_id exists in the local dev registry."""
    clean_id = (exercise_id or "").strip()
    return clean_id in _EXERCISES


def get_exercise(exercise_id: str) -> Exercise:
    """
    Retrieves an immutable Exercise from the static catalogue by its unique identifier.

    Raises:
        ExerciseNotFoundError: If the exercise_id is not found in the registry.
    """
    clean_id = (exercise_id or "").strip()
    if clean_id not in _EXERCISES:
        raise ExerciseNotFoundError(
            f"No exercise found for identifier '{exercise_id}'. "
            f"Available exercises: {list(_EXERCISES.keys())}."
        )
    return _EXERCISES[clean_id]


def get_all_exercises() -> Tuple[Exercise, ...]:
    """
    Returns an immutable tuple of all verified exercises in the registry.
    """
    return tuple(_EXERCISES.values())


def get_exercises_by_category(category: ExerciseCategory) -> Tuple[Exercise, ...]:
    """
    Returns all verified exercises belonging to the specified ExerciseCategory.
    """
    if not isinstance(category, ExerciseCategory):
        raise ValueError(f"Expected ExerciseCategory enum member, got {category}.")
    return tuple(ex for ex in _EXERCISES.values() if ex.category == category)


def get_available_exercise_ids() -> Tuple[str, ...]:
    """
    Returns an immutable tuple of all registered exercise identifiers.
    """
    return tuple(_EXERCISES.keys())
