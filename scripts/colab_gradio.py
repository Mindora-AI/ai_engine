#!/usr/bin/env python3
"""Gradio UI for MWIZERWA using ConversationEngine + TTSEngine + ASREngine."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.conversation.engine import ConversationEngine
from src.conversation.session import TurnRequest
from src.tts.tts_engine import TTSEngine, UnsupportedTTSLanguageError, TTSSynthesisError
from src.asr.asr_engine import ASREngine


class MWIZERWAService:
    def __init__(self, speaker_id: int = 0):
        self.engine = ConversationEngine()
        self.tts = TTSEngine(default_speaker=speaker_id)
        self.asr = ASREngine()

    def reset(self):
        self.engine.reset_conversation()
        return "Conversation reset.", None

    def chat_text(self, user_text: str, history: list):
        if not user_text or not user_text.strip():
            return history, "", None
        try:
            result = self.engine.process_turn(TurnRequest(user_text=user_text.strip()))
            reply = result.response_text
            history = history + [(user_text, reply)]
            audio_path = None
            try:
                audio_path = self.tts.synthesize_speech(reply)
            except (UnsupportedTTSLanguageError, TTSSynthesisError) as e:
                reply = f"{reply}\n\n[TTS Error] {e}"
            return history, "", audio_path
        except Exception as e:
            history = history + [(user_text, f"[Error] {e}")]
            return history, "", None

    def voice_turn(self, audio_path: str | None):
        if not audio_path:
            return "No audio provided.", None
        try:
            transcript = self.asr.transcribe_audio(audio_path)
            if not transcript.strip():
                return "ASR returned empty transcription.", None
            result = self.engine.process_turn(TurnRequest(user_text=transcript))
            reply = result.response_text
            out_audio = self.tts.synthesize_speech(reply)
            return f"You said: {transcript}\n\nAssistant: {reply}", out_audio
        except Exception as e:
            return f"[Voice Error] {e}", None


def build_ui(speaker_id: int = 0) -> gr.Blocks:
    svc = MWIZERWAService(speaker_id=speaker_id)
    with gr.Blocks(title="MWIZERWA AI") as demo:
        gr.Markdown(
            "# MWIZERWA AI\nMultilingual companion (Kinyarwanda, English, French). "
            "Uses the production ConversationEngine and TTS router."
        )
        with gr.Tab("Text chat + voice reply"):
            chatbot = gr.Chatbot(label="Conversation", type="tuples")
            msg = gr.Textbox(label="Your message")
            audio_out = gr.Audio(label="Assistant voice", type="filepath")
            with gr.Row():
                send = gr.Button("Send")
                clear = gr.Button("Reset conversation")
            send.click(svc.chat_text, inputs=[msg, chatbot], outputs=[chatbot, msg, audio_out])
            msg.submit(svc.chat_text, inputs=[msg, chatbot], outputs=[chatbot, msg, audio_out])
            clear.click(svc.reset, outputs=[chatbot, audio_out])
        with gr.Tab("Voice in → voice out"):
            audio_in = gr.Audio(label="Your voice (WAV)", type="filepath")
            voice_text = gr.Textbox(label="Transcript & reply", lines=6)
            voice_audio = gr.Audio(label="Assistant voice", type="filepath")
            gr.Button("Transcribe & respond").click(
                svc.voice_turn, inputs=audio_in, outputs=[voice_text, voice_audio]
            )
    return demo


def main():
    speaker = int(__import__("os").environ.get("MWIZERWA_SPEAKER_ID", "0"))
    demo = build_ui(speaker_id=speaker)
    demo.launch(share=False)


if __name__ == "__main__":
    main()