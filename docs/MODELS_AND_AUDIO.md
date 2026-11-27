# MWIZERWA AI — Models and Audio Architecture Guide

This guide documents the model assets, loading mechanisms, audio pipelines, and audio file inventory of MWIZERWA AI.

---

## 1. Conversational Language Model (LLM)

### 1.1 Model Specification
- **Model Name:** `AfriqueQwen3.5-4B-Instruct-v1`
- **Format:** GGUF (4-bit Quantization: `Q4_K_M`)
- **File Size:** **2.86 GB** (`AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf`)
- **Default Path:** `models/afriqueqwen/AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf`
- **Runtime:** `llama-cpp-python`

### 1.2 Context Window and Budgeting
- **`n_ctx` (Context Window):** 4096 tokens.
- **`max_tokens` (Response Budget):** 512 tokens.
- **`safety_margin`:** 16 tokens.
- **Input Budget Formula:**  
  $$\text{budget} = \text{n\_ctx} - \text{max\_tokens} - \text{safety\_margin} = 4096 - 512 - 16 = 3568\text{ tokens}$$
- **Chat Template:** Jinja2 template natively embedded in the GGUF file.

### 1.3 CPU Thread Configuration
In `src/llm/afrique_qwen.py`, CPU thread allocation defaults to:
$$\text{threads} = \max(1, \text{os.cpu\_count()} - 1)$$
To determine optimal thread counts on your CPU, run:
```powershell
python scripts/benchmark_cpu_threads.py
```

---

## 2. Text-to-Speech Engines (TTS)

MWIZERWA uses **two** synthesis engines selected automatically from the **final assistant response language**:

| Response language | Engine | Voice / checkpoint |
| :--- | :--- | :--- |
| `rw` (Kinyarwanda) | KinyaFlex | `models/kinya-flex-tts/kinya_flex_tts_base_trained.pt` |
| `en` (English) | Pocket TTS | voice `anna` (HF weights on first use) |
| `fr` (French) | Pocket TTS | voice `estelle` (`MWIZERWA_POCKET_FR_VOICE` to override) |

Implementation: `src/tts/tts_router.py` (`TTSRouter`) exposed as `TTSEngine` in `src/tts/tts_engine.py`.

### 2.1 KinyaFlex (Kinyarwanda)
- **Architecture:** `FlexKinyaTTS` (Multi-Band iSTFT VITS2) via `DeepKIN-AgAI`.
- **Checkpoint:** `models/kinya-flex-tts/kinya_flex_tts_base_trained.pt` (~1.03 GB).
- **Audio format:** 24 kHz, 16-bit mono PCM WAV.
- **Speaker:** fixed internal ID `0` (not configurable via API).

### 2.2 Pocket TTS (English / French)
- **Runtime:** `pocket-tts` ([Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)).
- **English:** predefined voice `anna`.
- **French:** voice `estelle` (override with `MWIZERWA_POCKET_FR_VOICE` if needed).
- **Install:** `pip install pocket-tts` (in `requirements.txt`). Weights download from Hugging Face on first synthesis.

### 2.3 Routing and safety
1. `resolve_response_language()` runs `LanguageDetector` on the **assistant response text** (not the user message).
2. Reliable `rw` → KinyaFlex; reliable `en` / `fr` → Pocket TTS.
3. Uncertain or unsupported languages raise `UnsupportedTTSLanguageError` — audio is **not** sent to the wrong engine.
4. Default synthesis output is `outputs/tts_responses/latest_response.wav`. Pass `output_path` only for tests or custom locations.

---

## 3. Automatic Speech Recognition (ASR)

- **Model Identifier:** `ElvisTata2024/Kinyarwanda-Health-ASR`
- **Architecture:** Hugging Face `transformers` ASR pipeline (`automatic-speech-recognition`).
- **Input Requirements:** 16 kHz WAV audio (mono preferred).
- **Caching:** Automatically downloaded on first use and cached to `~/.cache/huggingface/hub/`.

---

## 4. Audio and Video Assets Inventory

The repository maintains specific categories of audio files:

### Category A: Required Source and Test Assets (DO NOT DELETE)
- **`test.wav` (2.1 MB):**  
  A verified 16 kHz Kinyarwanda speech sample ("Amakuru ki?"). Used by `scripts/test_asr.py` and `launch_menu.py` for quick end-to-end diagnostics.
- **`kinyarwanda.mp4` (2.3 MB):**  
  A reference multimedia sample preserved in the root directory for prospective audiovisual alignment tasks.

### Category B: Approved Reference & Benchmark Outputs (PRESERVED)
- **`outputs/tts_baseline_evaluation/` (54 WAV files):**  
  Baseline evaluation audio generated across 18 phonetically challenging sentences for all 3 speakers (`spk0_...wav` to `spk2_...wav`), corresponding to the committed `manifest.json`.
- **`outputs/tts_phase2_experiments/`:**  
  Phase 2 experimental audio studying elision handling and speaking speed adjustments (0.92x, 0.95x, 1.00x).
- **`outputs/prosody_reference_set/`:**  
  Manifests, candidate selection reports, and inspection summaries for prosody reference datasets.

### Category C: Temporary Generated Audio (SAFE TO CLEAN)
- **`output.wav`:**  
  Created at the repository root when running `--mode text-voice` or `--mode voice`. Overwritten on each consecutive voice turn. Can be safely deleted at any time.
- **`outputs/tts_responses/`:**  
  Directory reserved for transient session audio exports.

---

## 5. Cleaning Temporary Audio Safely

To safely remove transient audio without deleting required test fixtures or benchmark baselines:
```powershell
# Remove temporary output.wav if present
if (Test-Path output.wav) { Remove-Item output.wav }
```

Do **not** delete `test.wav` or `outputs/tts_baseline_evaluation/manifest.json`.
