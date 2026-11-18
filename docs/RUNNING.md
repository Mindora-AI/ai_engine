# MWIZERWA AI — Running MWIZERWA

This guide documents the runtime entry points, operating modes, CLI parameters, and interactive usage of MWIZERWA AI.

---

## 1. Application Entry Points

MWIZERWA provides two primary entry points:

1. **`launch_menu.py` (Interactive Launcher):**  
   A user-friendly, numeric console menu offering all operating modes and quick diagnostic tests.
2. **`run_assistant.py` (Command-Line Entry Point):**  
   The core configurable entry point supporting scriptable flags, custom audio inputs, and speaker selection.

---

## 2. Using the Interactive Menu (`launch_menu.py`)

Run the launcher from your activated virtual environment:
```powershell
python launch_menu.py
```

The menu displays:
```text
=================================================================
       MWIZERWA AI — Kinyarwanda Local AI Assistant
=================================================================
Select a communication mode to run:

  1. Text Chat (Text-to-Text)
     -> Type text in console, get text reply.

  2. Text-to-Voice
     -> Type text in console, hear Kinyarwanda voice reply (WAV).

  3. Voice-to-Text
     -> Input WAV audio -> ASR transcribes -> LLM text reply.

  4. Voice-to-Voice (Full Pipeline)
     -> Input WAV audio -> ASR -> LLM -> KinyaFlex TTS voice reply.

  5. Quick Test: Kinyarwanda Health ASR on 'test.wav'
  6. Quick Test: KinyaFlex TTS 3-Voice Synthesis
  0. Exit
=================================================================
Enter choice (0-6) >
```

Select any number (1-6) to start the respective mode.

---

## 3. Direct CLI Execution (`run_assistant.py`)

### 3.1 Command-Line Arguments

```text
usage: run_assistant.py [-h] [--mode {text,voice,text-voice,voice-text}]
                        [--audio-file AUDIO_FILE] [--speaker SPEAKER]

Options:
  -h, --help            Show this help message and exit.
  --mode {text,voice,text-voice,voice-text}
                        Operating mode:
                        'text'       : Console chat, no audio required. (Default)
                        'voice'      : Full pipeline: WAV audio in -> ASR -> LLM -> TTS -> WAV.
                        'text-voice' : Type text in console -> TTS synthesizes voice reply.
                        'voice-text' : WAV audio in -> ASR -> LLM text reply (TTS skipped).
  --audio-file AUDIO_FILE
                        Optional WAV file path for single-turn testing.
  --speaker SPEAKER     Speaker ID for KinyaFlex TTS:
                        0 = Female Voice 1 (default)
                        1 = Female Voice 2
                        2 = Male Voice
```

### 3.2 Operating Modes and Examples

#### Mode 1: Text Chat (`--mode text`)
Best for testing conversation, language detection, safety gates, and exercises without audio overhead.
```powershell
python run_assistant.py --mode text
```
- Type your message at `You > ` and press Enter.
- Responses appear under `Assistant > `.
- Type `/clear` to reset conversation history while preserving system directives.
- Type `exit` or press `Ctrl+C` to quit.

#### Mode 2: Text-to-Voice (`--mode text-voice`)
You type text; the assistant generates a conversational response and synthesizes it to speech.
```powershell
python run_assistant.py --mode text-voice --speaker 0
```
- Response language is detected on the **assistant reply**; KinyaFlex synthesizes `rw`, Piper synthesizes `en` and `fr`.
- WAV output is saved to `outputs/tts_responses/latest_response.wav` (updated each turn).

#### Mode 3: Voice-to-Voice Full Pipeline (`--mode voice`)
Full voice round-trip: ASR → AfriqueQwen → trilingual TTS router (KinyaFlex / Piper).
- **Single-turn file test:**
  ```powershell
  python run_assistant.py --mode voice --audio-file test.wav --speaker 0
  ```
- **Interactive batch voice mode:**
  ```powershell
  python run_assistant.py --mode voice
  ```
  Prompt will ask for paths to `.wav` files consecutively.

#### Mode 4: Voice-to-Text (`--mode voice-text`)
Transcribes input speech and outputs text without running speech synthesis.
```powershell
python run_assistant.py --mode voice-text --audio-file test.wav
```

---

## 4. In-Session Interactive Commands

While running in interactive modes, the following console commands are recognized:

| Command | Action |
| :--- | :--- |
| `/clear` or `/reset` | Atomically resets conversation history to initial system prompt; clears any active or offered exercises. |
| `exit`, `quit`, or `q` | Exits the assistant cleanly. |
| `Ctrl+C` | Gracefully interrupts and exits the running session. |

---

## 5. Identifying Successful Model Loading and Execution

### 5.1 LLM Loading Indicators
When `AfriqueQwenLLM` initializes, the console displays:
```text
Loading AfriqueQwen model from: models/afriqueqwen/AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf ...
AfriqueQwen model loaded successfully!
```
- Memory usage will increase by ~3 GB in task manager / `htop`.
- If the file is missing or corrupted, a `FileNotFoundError` or `ValueError` is printed immediately.

### 5.2 ASR Initialization Indicators
When `ASREngine` runs:
```text
[1/4] Transcribing audio with ASR: test.wav...
[2/4] Transcription output: '...'
```
- On first execution, the Hugging Face model (`ElvisTata2024/Kinyarwanda-Health-ASR`) is cached to `~/.cache/huggingface/hub/`.

### 5.3 TTS Synthesis Indicators
When `TTSEngine` runs:
```text
[TTS Router] Routing to Piper for language 'en'...
[Piper TTS] Synthesized audio saved to: ...\outputs\tts_responses\latest_response.wav
[TTS Audio Ready] Saved at: ...
```
- KinyaFlex output: 24 kHz mono PCM. Piper output: 22.05 kHz mono PCM (voice-dependent).
