"""
Interactive Launcher for MWIZERWA AI / Kinyarwanda Local Voice Assistant.
Provides a simple menu to launch any communication mode:
1. Text Chat (Text-to-Text)
2. Text-to-Voice (Type text -> KinyaFlex TTS voice reply)
3. Voice-to-Text (WAV audio -> ASR -> LLM text reply)
4. Voice-to-Voice (WAV audio -> ASR -> LLM -> KinyaFlex TTS audio reply)
5. Test Kinyarwanda Health ASR on test.wav
6. Test KinyaFlex TTS Multi-Speaker Synthesis
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_assistant


def print_banner():
    print("\n" + "=" * 65)
    print("       MWIZERWA AI — Kinyarwanda Local AI Assistant")
    print("=" * 65)
    print("Select a communication mode to run:\n")
    print("  1. Text Chat (Text-to-Text)")
    print("     -> Type text in console, get text reply.")
    print()
    print("  2. Text-to-Voice")
    print("     -> Type text in console, hear Kinyarwanda voice reply (WAV).")
    print()
    print("  3. Voice-to-Text")
    print("     -> Input WAV audio -> ASR transcribes -> LLM text reply.")
    print()
    print("  4. Voice-to-Voice (Full Pipeline)")
    print("     -> Input WAV audio -> ASR -> LLM -> KinyaFlex TTS voice reply.")
    print()
    print("  5. Quick Test: Kinyarwanda Health ASR on 'test.wav'")
    print("  6. Quick Test: KinyaFlex TTS 3-Voice Synthesis")
    print("  0. Exit")
    print("=" * 65)


def get_default_sample_audio():
    for name in ["kinyarwanda.wav", "test.wav"]:
        if (PROJECT_ROOT / name).exists():
            return name
    return "kinyarwanda.wav"


def main():
    while True:
        sample_audio = get_default_sample_audio()
        print_banner()
        choice = input("Enter choice (0-6) > ").strip()

        if choice in ["0", "exit", "quit", "q"]:
            print("\nMurakoze cyane! Bye!")
            break

        if choice == "1":
            print("\nLaunching Text Chat Mode...")
            manager = run_assistant.ConversationManager()
            run_assistant.run_text_mode(manager)

        elif choice == "2":
            print("\nSelect Voice / Speaker for KinyaFlex TTS:")
            print("  0. Female Voice 1 (default)")
            print("  1. Female Voice 2")
            print("  2. Male Voice")
            spk_input = input("Speaker ID [0/1/2, default: 0] > ").strip()
            speaker = int(spk_input) if spk_input in ["0", "1", "2"] else 0
            manager = run_assistant.ConversationManager()
            run_assistant.run_text_voice_mode(manager, speaker_id=speaker)

        elif choice == "3":
            print(f"\nDo you want to test with the existing '{sample_audio}' or enter a path?")
            print(f"  1. Use existing sample: '{sample_audio}'")
            print("  2. Interactive mode (prompt for file path)")
            sub = input("Choice [1/2, default: 1] > ").strip()
            audio_file = sample_audio if sub != "2" else None
            manager = run_assistant.ConversationManager()
            run_assistant.run_voice_text_mode(manager, audio_file=audio_file)

        elif choice == "4":
            print("\nSelect Voice / Speaker for KinyaFlex TTS reply:")
            print("  0. Female Voice 1 (default)")
            print("  1. Female Voice 2")
            print("  2. Male Voice")
            spk_input = input("Speaker ID [0/1/2, default: 0] > ").strip()
            speaker = int(spk_input) if spk_input in ["0", "1", "2"] else 0

            print(f"\nDo you want to test with the existing '{sample_audio}' or enter a path?")
            print(f"  1. Use existing sample: '{sample_audio}'")
            print("  2. Interactive mode (prompt for file path)")
            sub = input("Choice [1/2, default: 1] > ").strip()
            audio_file = sample_audio if sub != "2" else None
            manager = run_assistant.ConversationManager()
            run_assistant.run_voice_mode(manager, audio_file=audio_file, speaker_id=speaker)

        elif choice == "5":
            print(f"\nRunning test_asr.py on '{sample_audio}'...")
            import subprocess
            subprocess.run([sys.executable, "scripts/test_asr.py"])

        elif choice == "6":
            print("\nRunning test_kinya_flex.py (generating 3 speaker samples)...")
            import subprocess
            subprocess.run([sys.executable, "scripts/test_kinya_flex.py"])

        else:
            print("[Warning] Invalid selection. Please enter 0 to 6.")


if __name__ == "__main__":
    main()
