# MWIZERWA AI — Isolated engine API test guide

Hands-on verification of every public HTTP endpoint **after** the API server is running with models available.

| Reference | Use |
|-----------|-----|
| [docs/API.md](API.md) | Field definitions |
| [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) §8 | Backend integration |
| `src/api/schemas.py` | Canonical schemas |
| Swagger UI | Interactive try-it-out |

---

## 0. Prerequisites

### What you need

| Requirement | Notes |
|-------------|--------|
| Python **3.10–3.12** | Same venv as the project |
| **AfriqueQwen GGUF** | Under `models/afriqueqwen/` (see `scripts/download_models.py`) |
| Core deps | `requirements.txt` |
| API deps | `requirements-api.txt` |
| Optional EN/FR ASR | `requirements-asr-whisper.txt` for `language=en` or `fr` on `/transcribe` and `/voice` |
| KinyaFlex + Pocket TTS | TTS for `/tts` and `/voice` reply audio |

### Start the server (local)

**Windows (PowerShell):**

```powershell
cd path\to\ai_engine
.\.venv\Scripts\activate
$env:MWIZERWA_ENV = "development"   # optional: enables POST /memory/consent
python scripts/serve_api.py
```

**Linux / macOS:**

```bash
cd path/to/ai_engine
source .venv/bin/activate
export MWIZERWA_ENV=development
python scripts/serve_api.py
```

Default base URL: **`http://127.0.0.1:8000`**

Override bind:

```powershell
$env:MWIZERWA_API_HOST = "0.0.0.0"
$env:MWIZERWA_API_PORT = "8000"
```

Point models (if not default `models/`):

```powershell
$env:MWIZERWA_MODELS_DIR = "D:\path\to\models"
```

### Wait until ready

The LLM loads at startup (can take **1–3 minutes** on CPU, less on GPU).

```powershell
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Proceed when `"engine_initialized": true` and `"status": "ready"`.

### Swagger UI (all endpoints)

1. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Expand an operation → **Try it out** → fill parameters → **Execute**
3. For file uploads (`/transcribe`, `/voice`), use **Choose file**

OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### Colab / remote

Run the **HTTP API** cells in `notebooks/MWIZERWA_Colab.ipynb` (after Drive + models), or follow [COLAB.md](COLAB.md). Use ngrok public URL instead of `127.0.0.1` for tests from your laptop. Ngrok free tier may require the `ngrok-skip-browser-warning: true` header.

Set `BASE` in examples below:

```powershell
$env:MWIZERWA_API_BASE = "http://127.0.0.1:8000"
```

```bash
export MWIZERWA_API_BASE=http://127.0.0.1:8000
```

---

## 1. Endpoint checklist

| # | Method | Path | Needs LLM | Needs audio file | Optional deps |
|---|--------|------|:-----------:|:----------------:|---------------|
| 1 | GET | `/health` | No (reports load state) | No | — |
| 2 | POST | `/chat` | **Yes** | No | — |
| 3 | POST | `/tts` | No | No | TTS weights |
| 4 | POST | `/transcribe` | No | **Yes** | HF ASR; Whisper for en/fr |
| 5 | POST | `/voice` | **Yes** | **Yes** | ASR + TTS |
| 6 | POST | `/memory/consent` | No | No | Dev env only (see §6) |

---

## 2. `GET /health`

**Purpose:** Readiness without loading ASR/TTS on first call.

### Terminal

```powershell
curl -s "%MWIZERWA_API_BASE%/health"
```

### Expected success (shape)

```json
{
  "status": "ready",
  "engine_initialized": true,
  "backends": {
    "llm": { "dependency_available": true, "artifact_present": true, "loaded": true },
    "asr_kinyarwanda": { "dependency_available": true, "loaded": false },
    "asr_whisper": { "dependency_available": true, "configured_model": "small", "loaded": false },
    "tts": { "dependency_available": true, "artifact_present": true, "loaded": false }
  }
}
```

### Swagger

- **GET /health** → Execute (no body)

### Pass criteria

- HTTP **200**
- If server just started: `engine_initialized` becomes `true` after GGUF load
- `backends.asr_whisper.dependency_available` is `false` if `faster-whisper` not installed

### Failure cases

| Observation | Meaning |
|-------------|---------|
| `status: "unavailable"`, `engine_initialized: false` | GGUF missing or failed load — check console logs |
| Connection refused | `serve_api.py` not running |

---

## 3. `POST /chat`

**Purpose:** Full text turn (safety, conversation, optional context, recommendations).

**Content-Type:** `application/json`

### 3.1 Minimal chat (terminal)

**PowerShell:**

```powershell
$body = @{ user_text = "Muraho, umeze ute?" } | ConvertTo-Json
Invoke-RestMethod -Uri "$env:MWIZERWA_API_BASE/chat" -Method Post -Body $body -ContentType "application/json"
```

**curl (bash / Git Bash):**

```bash
curl -s -X POST "$MWIZERWA_API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_text":"Hello, how are you today?","language":"en"}'
```

### 3.2 Session continuity

Use `session_id` from the first response on the second call:

```json
{
  "user_text": "Can you remember what I just said?",
  "session_id": "paste-session-id-here"
}
```

### 3.3 Backend-style context

```json
{
  "user_text": "I feel tense.",
  "session_id": "manual-test-1",
  "language": "en",
  "context": {
    "approved_memory_summary": "User prefers evening check-ins.",
    "recent_turns": [
      { "role": "user", "content": "Work was stressful." },
      { "role": "assistant", "content": "I hear how heavy that felt." }
    ],
    "eligible_exercises": [
      { "id": "breathing_gentle_rhythm", "category": "breathing" }
    ]
  }
}
```

### 3.4 Optional `user_id` (memory proposals)

Requires dev memory path; proposals appear in `metadata.memory_proposals` when the extractor fires:

```json
{
  "user_text": "I usually unwind by journaling at night.",
  "user_id": "test-user-trusted-backend-only",
  "session_id": "mem-test-1"
}
```

### Swagger

1. **POST /chat** → Try it out
2. Paste JSON into Request body
3. Execute (allow **up to 2 minutes** for first inference on CPU)

### Pass criteria

- HTTP **200**
- Non-empty `response_text`
- `session_id` returned
- `metadata` contains only public keys (e.g. `gate1_triggered`, `detected_language`)

### Negative tests

| Request | Expected |
|---------|----------|
| `{"user_text": "   "}` | **422** validation |
| `{"user_text": "Hi", "language": "de"}` | **422** |
| Call before LLM loaded | **503** |

---

## 4. `POST /tts`

**Purpose:** Synthesize speech to **WAV bytes** (not JSON).

**Content-Type:** `multipart/form-data`

### Terminal

```powershell
curl -s -X POST "%MWIZERWA_API_BASE%/tts" `
  -F "text=Hello from MWIZERWA" `
  -F "language=en" `
  -o smoke_tts_en.wav
```

```powershell
curl -s -X POST "%MWIZERWA_API_BASE%/tts" `
  -F "text=Muraho" `
  -F "language=rw" `
  -o smoke_tts_rw.wav
```

### Swagger

1. **POST /tts**
2. `text` = required string
3. Optional `language` (`rw` / `en` / `fr`)
4. Execute → **Download** response body

### Pass criteria

- HTTP **200**
- `Content-Type` includes `audio/wav`
- File plays in a media player

### Negative tests

| Case | Expected |
|------|----------|
| Empty `text` | **400** |

---

## 5. `POST /transcribe`

**Purpose:** ASR only.

| `language` form field | Engine |
|----------------------|--------|
| omitted or `rw` | Kinyarwanda Health ASR |
| `en` or `fr` | Whisper Small (`faster-whisper`) |

### Prepare audio

- Record a WAV or use TTS output from §4
- Synthetic benchmark clips: run `python scripts/prepare_asr_benchmark_clips.py` (local, gitignored WAVs)

### Terminal

```powershell
curl -s -X POST "%MWIZERWA_API_BASE%/transcribe" `
  -F "audio=@smoke_tts_en.wav" `
  -F "language=en"
```

### Swagger

- Upload file under `audio`
- Optional `language`: `rw` | `en` | `fr`

### Pass criteria

- HTTP **200**
- JSON: `{ "transcription": "...", "language": "en" }`
- First `rw` call may take longer (HF model load)

### Negative tests

| Case | Expected |
|------|----------|
| `language=fr` without faster-whisper | **503** |
| Oversized file | **413** (limit `MWIZERWA_MAX_UPLOAD_BYTES`, default 25 MiB) |

---

## 6. `POST /memory/consent` (development only)

**Not** a production endpoint. Enabled when:

- `MWIZERWA_ENV=development` (or `dev`, `test`, `local`), **or**
- `MWIZERWA_ALLOW_DEV_MEMORY_CONSENT=1`

Otherwise returns **404**.

### Workflow

1. `POST /chat` with `user_id` until `metadata.memory_proposals` contains an entry
2. Note `proposal_id` from the proposal object
3. Consent:

```json
{
  "user_id": "test-user-trusted-backend-only",
  "proposal_id": "paste-proposal-id",
  "decision": "accepted"
}
```

```powershell
$body = @{
  user_id = "test-user-trusted-backend-only"
  proposal_id = "YOUR_PROPOSAL_ID"
  decision = "accepted"
} | ConvertTo-Json
Invoke-RestMethod -Uri "$env:MWIZERWA_API_BASE/memory/consent" -Method Post -Body $body -ContentType "application/json"
```

### Pass criteria

- HTTP **200** with `applied: true` when accepted
- Later `/chat` with same `user_id` may reflect merged memory (dev in-memory store)

---

## 7. `POST /voice`

**Purpose:** ASR → chat → TTS in one request. **Does not accept `user_id`.**

**Content-Type:** `multipart/form-data`

### Terminal

```powershell
curl -s -X POST "%MWIZERWA_API_BASE%/voice" `
  -F "audio=@smoke_tts_en.wav" `
  -F "language=en" `
  -F "session_id=voice-test-1" `
  | python -m json.tool
```

### Optional `context_json`

Single form field containing stringified `TurnContextRequest` JSON:

```powershell
$ctx = '{"eligible_exercises":[{"id":"breathing_gentle_rhythm","category":"breathing"}]}'
curl -s -X POST "%MWIZERWA_API_BASE%/voice" `
  -F "audio=@smoke_tts_en.wav" `
  -F "context_json=$ctx"
```

### Swagger

- Upload `audio`
- Fill optional fields; for `context_json`, paste minified JSON as **string**

### Pass criteria

- HTTP **200**
- `transcription` and `response_text` present
- `audio_status`: `"success"` or `"failed"`
- If `"failed"`: `audio_base64` is `null`, `audio_error` explains; **do not** retry same audio to avoid double conversation advance

### Decode reply audio (Python)

```python
import base64, json, requests

r = requests.post(
    "http://127.0.0.1:8000/voice",
    files={"audio": open("smoke_tts_en.wav", "rb")},
    data={"language": "en"},
    timeout=300,
)
data = r.json()
if data.get("audio_status") == "success" and data.get("audio_base64"):
    open("reply.wav", "wb").write(base64.b64decode(data["audio_base64"]))
```

### Memory-aware voice (integration pattern)

1. `POST /transcribe` with audio  
2. `POST /chat` with transcript + trusted `user_id` + `context`  
3. Optional `POST /tts` for spoken reply  

See INTEGRATION_GUIDE §8.3.

---

## 8. Suggested test order (manual QA)

| Step | Action | Confirms |
|------|--------|----------|
| 1 | `GET /health` until ready | LLM + artifacts |
| 2 | `POST /chat` minimal | Inference pipeline |
| 3 | `POST /chat` same `session_id` | In-memory session |
| 4 | `POST /chat` with `context` | Backend context contract |
| 5 | `POST /tts` en + rw | TTS routing |
| 6 | `POST /transcribe` rw | Kinya ASR |
| 7 | `POST /transcribe` en (if Whisper installed) | EN ASR |
| 8 | `POST /voice` | End-to-end voice |
| 9 | `POST /memory/consent` (dev env) | Dev memory loop |

**Typical duration:** 15–45 minutes including model load and first ASR/TTS cold starts.

---

## 9. Automated regression (no manual server)

Does **not** replace live endpoint testing; validates contracts with mocks:

```powershell
.\.venv\Scripts\activate
pytest tests/test_api.py tests/test_integration_contracts.py -q
```

Full suite:

```powershell
pytest tests/ -q
```

---

## 10. Troubleshooting

| Symptom | Check |
|---------|--------|
| **503** on `/chat` | `/health` → `engine_initialized` |
| **503** EN/FR transcribe | `pip install -r requirements-asr-whisper.txt` |
| **503** TTS | KinyaFlex checkpoint under `MWIZERWA_MODELS_DIR`; `pip install pocket-tts` for EN/FR |
| Slow first turn | Normal: GGUF + cold ASR/TTS |
| Swagger upload fails | File under 25 MiB; use WAV |
| **404** `/memory/consent` | Set `MWIZERWA_ENV=development` |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 11. What this guide does not cover

- Application backend authentication or production memory DB
- Load testing or multi-worker session affinity
- Natural conversational ASR accuracy sign-off (see `benchmarks/asr_en_fr/`)
- Frontend or E2E product tests

For backend integration acceptance criteria, see INTEGRATION_GUIDE §8.11.