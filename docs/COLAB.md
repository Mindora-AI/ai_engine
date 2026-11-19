# MWIZERWA AI — Google Colab deployment guide

**Repository:** [Mindora-AI/ai_engine](https://github.com/Mindora-AI/ai_engine)  
**Notebook:** `notebooks/MWIZERWA_Colab.ipynb`  
**Colab checkout path:** `/content/ai_engine`

## 1. Open the notebook

1. Sync this repository to GitHub (`Mindora-AI/ai_engine`).
2. Open `notebooks/MWIZERWA_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
3. **Runtime → Change runtime type → GPU** (T4 or better recommended; High-RAM if possible).

Cell 1 clones `ai_engine` when `/content/ai_engine` is missing and verifies `scripts/colab_runtime.py` exists.  
If those files are missing on GitHub, push the latest `main` before running Colab.

## 2. Google Drive layout

Persistent storage (survives runtime disconnect):

```text
/content/drive/MyDrive/MWIZERWA/
├── models/
│   ├── afriqueqwen/
│   │   └── AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf
│   ├── kinya-flex-tts/
│   │   └── kinya_flex_tts_base_trained.pt
│   └── piper/
│       ├── en_US-ljspeech-medium/
│       └── fr_FR-siwis-medium/
├── cache/              # HF hub cache (HF_HOME)
├── outputs/            # TTS + validation WAVs
└── logs/
```

Ephemeral VM source checkout:

```text
/content/ai_engine/     # git clone of ai_engine (not stored on Drive)
```

## 3. Environment variables

| Variable | Purpose |
|----------|---------|
| `MWIZERWA_PROJECT_ROOT` | `/content/ai_engine` |
| `MWIZERWA_MODELS_DIR` | `MyDrive/MWIZERWA/models` |
| `MWIZERWA_CACHE_DIR` | `MyDrive/MWIZERWA/cache` |
| `MWIZERWA_OUTPUT_DIR` | `MyDrive/MWIZERWA/outputs` |
| `MWIZERWA_LLM_N_GPU_LAYERS` | llama-cpp GPU layers (0 = CPU LLM) |
| `MWIZERWA_LLM_MAX_TOKENS` | Completion budget (default 768) |
| `HF_HOME` | `MyDrive/MWIZERWA/cache/huggingface` (ASR weights) |
| `MWIZERWA_REPO_URL` | Optional clone URL override |
| `HF_TOKEN` | Optional Hugging Face token (Colab Secrets) |

Set by `scripts/colab_runtime.configure_drive_persistence()`.

## 4. Setup sequence (notebook order)

| Step | Action |
|------|--------|
| 1 | Clone / reuse `/content/ai_engine`, `require_colab_scripts()` |
| 2 | `drive.mount` → `configure_drive_persistence()` |
| 3 | `inspect_hardware()` |
| 4 | `install_python_dependencies()` — **does not** reinstall Colab `torch` |
| 5 | `install_llama_cpp_for_colab()` + `apply_llm_gpu_layers()` |
| 6 | `install_deepkin()` + `verify_critical_imports()` |
| 7 | `download_models.py --check` → download → `--check` again |
| 8 | `colab_validate.py` (exit 0 required) |
| 9 | `colab_gradio.py` only if validation passed |

## 5. Dependencies

- `requirements-colab.txt` → `requirements-colab-packages.txt` (no `torch` / `llama-cpp-python`).
- Colab keeps its CUDA-enabled PyTorch.
- `llama-cpp-python` is selected by `install_llama_cpp_for_colab()` using `torch.version.cuda` and verified with `llama_supports_gpu_offload()`.

## 6. Model registry

Authoritative metadata: `config/model_registry.json`  
Downloader: `python scripts/download_models.py --check` | `python scripts/download_models.py`

Do not pass `|| true` on `--check` — missing models must fail visibly.

## 7. Application pipeline (unchanged)

The notebook uses production code paths:

- `ConversationEngine` (Gate 1/2, exercises, language detection)
- `AfriqueQwenLLM` (GGUF)
- `TTSEngine` / `TTSRouter` — KinyaFlex (`rw`), Piper (`en`, `fr`) on **assistant response** text
- `ASREngine` (voice tab and optional `test.wav`)
- `scripts/colab_gradio.py` — same stack as `run_assistant.py`

## 8. Validation

| Command | What it checks |
|---------|----------------|
| `python scripts/verify_notebook_imports.py` | Notebook imports vs `colab_runtime.py` (static) |
| `python scripts/download_models.py --check` | Files + checksums on Drive |
| `python scripts/colab_validate.py` | Real LLM/TTS + metadata + Gradio UI build |
| `pytest tests/test_colab_runtime.py` | Runtime helpers (mocked / local) |
| `pytest tests/test_tts_routing.py` | TTS routing (mocked engines) |

`colab_validate.py` distinguishes required vs optional (`asr_test_wav`, `voice_to_voice`).

## 9. Troubleshooting

| Symptom | Action |
|---------|--------|
| `cannot import name 'HfFolder' from 'huggingface_hub'` | Re-run deps cell — installs `gradio>=5.7.1` (incompatible with Gradio 4 + hub 1.x) |
| `No module named 'deepkin'` | Re-run DeepKIN cell; check git clone of `c4ir-rw/ac-ai-models` and pip `-e DeepKIN-AgAI --no-deps` |
| `colab_runtime.py` not found after clone | Push latest `ai_engine` to GitHub; re-run cell 1 |
| Drive mount cancelled | Re-run Drive cell; do not continue |
| `No compatible CUDA wheel` | CPU LLM (`N_GPU_LAYERS=0`); inference is slower |
| OOM on GPU | Lower `MWIZERWA_LLM_N_GPU_LAYERS` or use High-RAM |
| Missing models | `download_models.py`; confirm `MWIZERWA_MODELS_DIR` on Drive |
| KinyaFlex import error | Re-run DeepKIN cell |
| Validation fails at `gradio_ui_build` | Fix prior LLM/TTS errors; check `gradio` install |

## 10. Pre-flight checklist

- [ ] Drive mounted; all four env paths printed
- [ ] `download_models.py --check` all required assets OK
- [ ] `llama_supports_gpu_offload()` matches expectations
- [ ] `colab_validate.py` passes (rw/en/fr LLM + TTS + routing)
- [ ] Gradio launches after validation flag
- [ ] Second session reuses Drive models (no redundant GGUF download)

See also [COLAB_AUDIT.md](COLAB_AUDIT.md), [INSTALLATION.md](INSTALLATION.md), [MODELS_AND_AUDIO.md](MODELS_AND_AUDIO.md).