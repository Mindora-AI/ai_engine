import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure src is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.conversation import ConversationManager
from src.asr.asr_engine import ASREngine
from src.tts.tts_engine import TTSEngine


def run_text_mode(manager: ConversationManager):
    """
    Runs text chat mode in console.
    Independent of ASR and TTS.
    """
    print("\n" + "=" * 60)
    print("  Kinyarwanda Local Voice Assistant - Text Chat Mode")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("Commands:")
    print("  /clear  - Reset conversation history")
    print("  exit    - Quit the assistant\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting assistant. Murakoze!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting assistant. Murakoze!")
            break

        if user_input.lower() in ["/clear", "/reset"]:
            manager.reset_conversation()
            print("[System] Conversation history reset (System prompt preserved).\n")
            continue

        try:
            response = manager.generate_response(user_input)
            print(f"\nAssistant > {response}\n")
        except Exception as e:
            print(f"\n[Error] Failed to generate response: {e}\n")


def run_text_voice_mode(
    manager: ConversationManager,
    speaker_id: int = 0,
):
    """
    Runs text-to-voice mode:
    User types text -> ConversationManager -> AfriqueQwen LLM -> KinyaFlex TTS -> WAV Output.
    ASR is skipped entirely.
    """
    tts = TTSEngine(default_speaker=speaker_id)

    print("\n" + "=" * 60)
    print("  Kinyarwanda Local Voice Assistant - Text-to-Voice Mode")
    print("=" * 60)
    print("Type your message and press Enter. The assistant replies in voice.")
    print("Commands:")
    print("  /clear  - Reset conversation history")
    print("  exit    - Quit the assistant\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting assistant. Murakoze!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting assistant. Murakoze!")
            break

        if user_input.lower() in ["/clear", "/reset"]:
            manager.reset_conversation()
            print("[System] Conversation history reset (System prompt preserved).\n")
            continue

        # Step 1: Generate LLM response
        print("[1/2] Generating response with AfriqueQwen...")
        try:
            response = manager.generate_response(user_input)
            print(f"\nAssistant > {response}\n")
        except Exception as e:
            print(f"\n[LLM Error] Failed to generate response: {e}\n")
            continue

        if not response or not response.strip():
            print("[LLM Warning] Empty response. Skipping TTS.\n")
            continue

        # Step 2: Synthesize speech
        print("[2/2] Synthesizing speech with KinyaFlex TTS...")
        try:
            audio_output = tts.synthesize_speech(response)
            if audio_output:
                print(f"[TTS Audio Ready] Saved at: {audio_output}\n")
            else:
                print("[TTS Info] Speech output skipped (response not in Kinyarwanda or empty).\n")
        except Exception as e:
            print(f"[TTS Error] Speech synthesis failed: {e}\n")


def run_voice_mode(
    manager: ConversationManager,
    audio_file: str = None,
    speaker_id: int = 0,
):
    """
    Runs voice mode pipeline:
    Audio Input -> ASR -> Conversation Manager -> AfriqueQwen -> Response Cleaner -> KinyaFlex TTS -> WAV Output
    """
    print("\n" + "=" * 60)
    print("  Kinyarwanda Local Voice Assistant - Voice Mode")
    print("=" * 60)

    # Initialize ASR and TTS engines lazily
    asr = ASREngine()
    tts = TTSEngine(default_speaker=speaker_id)

    if audio_file:
        print(f"Processing audio file: {audio_file}")
        transcribe_and_respond(manager, asr, tts, audio_file)
        return

    print("Voice mode activated.")
    print("Type path to a WAV audio file, or type '/clear' to reset, 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Voice Input (WAV file path or command) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting voice assistant. Murakoze!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting voice assistant. Murakoze!")
            break

        if user_input.lower() in ["/clear", "/reset"]:
            manager.reset_conversation()
            print("[System] Conversation history reset.\n")
            continue

        wav_path = Path(user_input)
        if not wav_path.exists():
            print(f"[Error] Audio file not found: {wav_path.resolve()}\n")
            continue

        transcribe_and_respond(manager, asr, tts, str(wav_path))


def run_voice_text_mode(
    manager: ConversationManager,
    audio_file: str = None,
):
    """
    Runs voice-to-text mode:
    Audio Input (WAV) -> ASR -> Conversation Manager -> AfriqueQwen -> Response Cleaner -> Text Output.
    TTS speech synthesis is skipped.
    """
    print("\n" + "=" * 60)
    print("  Kinyarwanda Local Voice Assistant - Voice-to-Text Mode")
    print("=" * 60)

    asr = ASREngine()

    if audio_file:
        print(f"Processing audio file: {audio_file}")
        transcribe_and_respond(manager, asr, None, audio_file)
        return

    print("Voice-to-Text mode activated.")
    print("Type path to a WAV audio file, or type '/clear' to reset, 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Voice Input (WAV file path or command) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting assistant. Murakoze!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting assistant. Murakoze!")
            break

        if user_input.lower() in ["/clear", "/reset"]:
            manager.reset_conversation()
            print("[System] Conversation history reset.\n")
            continue

        wav_path = Path(user_input)
        if not wav_path.exists():
            print(f"[Error] Audio file not found: {wav_path.resolve()}\n")
            continue

        transcribe_and_respond(manager, asr, None, str(wav_path))


def transcribe_and_respond(
    manager: ConversationManager,
    asr: ASREngine,
    tts: Optional[TTSEngine],
    audio_path: str,
):
    """
    Processes a single voice turn from an audio WAV file.
    """
    print(f"\n[1/4] Transcribing audio with ASR: {audio_path}...")
    try:
        transcription = asr.transcribe_audio(audio_path)
    except Exception as e:
        print(f"[ASR Error] Transcription failed: {e}\n")
        return

    if not transcription or not transcription.strip():
        print("[ASR Warning] Transcription empty or silent. Skipping LLM generation.\n")
        return

    print(f"[2/4] Transcription output: '{transcription}'")

    print("[3/4] Generating response with AfriqueQwen...")
    try:
        response = manager.generate_response(transcription)
        print(f"\nAssistant > {response}\n")
    except Exception as e:
        print(f"[LLM Error] Generation failed: {e}\n")
        return

    if not response or not response.strip():
        print("[LLM Warning] Empty response generated. Skipping TTS.\n")
        return

    if tts is not None:
        print("[4/4] Synthesizing speech with KinyaFlex TTS...")
        try:
            audio_output = tts.synthesize_speech(response)
            if audio_output:
                print(f"[TTS Audio Ready] Saved at: {audio_output}\n")
            else:
                print("[TTS Info] Speech output skipped (language non-Kinyarwanda or empty).\n")
        except Exception as e:
            print(f"[TTS Error] Speech synthesis failed: {e}\n")
    else:
        print("[Voice-to-Text] Turn complete. (Speech synthesis skipped in this mode).\n")


def main():
    parser = argparse.ArgumentParser(
        description="Local Voice Assistant (ASR + AfriqueQwen LLM + KinyaFlex TTS)"
    )
    parser.add_argument(
        "--mode",
        choices=["text", "voice", "text-voice", "voice-text"],
        default="text",
        help=(
            "Operating mode: "
            "'text' (console chat, no audio), "
            "'voice' (ASR+TTS full pipeline: voice in -> voice out), "
            "'text-voice' (type text -> voice reply), or "
            "'voice-text' (voice in -> text reply). "
            "Default: text"
        ),
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Optional WAV file path for single-turn voice mode testing.",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID for KinyaFlex TTS (0=female_1, 1=female_2, 2=male). Default: 0",
    )

    args = parser.parse_args()

    manager = ConversationManager()

    if args.mode == "text":
        run_text_mode(manager)
    elif args.mode == "voice":
        run_voice_mode(manager, audio_file=args.audio_file, speaker_id=args.speaker)
    elif args.mode == "text-voice":
        run_text_voice_mode(manager, speaker_id=args.speaker)
    elif args.mode == "voice-text":
        run_voice_text_mode(manager, audio_file=args.audio_file)


if __name__ == "__main__":
    main()
