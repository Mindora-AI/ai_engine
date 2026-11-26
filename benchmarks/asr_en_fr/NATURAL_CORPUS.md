# Natural EN/FR speech (manual acquisition)

Synthetic Piper clips in `manifest.json` are **not** conversational ground truth. Add licensed natural clips with `sample_type: "natural"`.

## Required manifest fields per natural clip

| Field | Requirement |
|--------|-------------|
| `id` | Unique stable id (e.g. `en_nat_cv_01`) |
| `language` | `en` or `fr` |
| `sample_type` | `natural` |
| `file` | Path under this directory (e.g. `clips/natural/en_nat_cv_01.wav`) |
| `reference_text` | Human-verified transcript (exact match protocol for WER) |
| `source` | Corpus + clip id (e.g. `mozilla-common-voice`, split, clip id) |
| `license` | SPDX or URL (e.g. `CC0-1.0`) |
| `notes` | Recording conditions, speaker metadata if allowed |

Synthetic clips use `sample_type: "synthetic"` and Piper `source`; natural clips **must** include `license`.

## Smallest practical path (recommended)

1. [Mozilla Common Voice](https://commonvoice.mozilla.org/) — CC0, EN + FR.
2. Download a **small validation subset** (2–4 clips per language, ~3–10 s).
3. Export WAV (16 kHz mono preferred; resample if needed).
4. Copy official sentence text as `reference_text` from the dataset metadata (do not paraphrase).
5. Append entries to `manifest.json` and run `benchmark_whisper_asr.py`.

## Hugging Face access

Common Voice on the Hub often requires:

1. Log in: `huggingface-cli login`
2. Accept the dataset terms on the model card for `mozilla-foundation/common_voice_17_0`
3. Optional: `HF_TOKEN` for reliable downloads

Then run:

```powershell
pip install datasets soundfile
.\.venv\Scripts\python.exe scripts/prepare_natural_cv_clips.py --max-clips 25
```

## Status

**Pending:** Automated fetch failed in this environment (`DatasetNotFoundError` / empty Hub tree without authenticated access). Schema, script, and manual steps are ready; natural-speech **WER validation is not complete** until clips are acquired.