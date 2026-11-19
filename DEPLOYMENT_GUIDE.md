# MWIZERWA — Developer deployment and integration guide

**Repository:** [Mindora-AI/ai_engine](https://github.com/Mindora-AI/ai_engine)

This guide is for developers building apps that call the **MWIZERWA inference HTTP API** when it is run from **Google Colab** and exposed via a **temporary ngrok tunnel**.

It is **not** a guarantee of production uptime. Treat the ngrok URL as **ephemeral**.

---

## 1. Architecture

```text
Your application
    → HTTPS public ngrok URL
        → FastAPI (scripts/serve_api.py) in Colab
            → ConversationEngine (AfriqueQwen GGUF)
            → ASREngine (Hugging Face ASR)
            → TTSEngine (KinyaFlex + Piper)
```

| Component | Role |
|-----------|------|
| **Colab runtime** | Hosts Python, GPU, and loaded models |
| **FastAPI** | `GET /health`, `POST /chat`, `/tts`, `/transcribe`, `/voice` |
| **ngrok** | Public HTTPS forward to Colab port (e.g. 8000) |

**Operational facts:**

- The API runs only while the **Colab notebook/runtime** and **uvicorn process** stay up.
- The **ngrok URL changes** whenever you restart the tunnel (unless you use a reserved ngrok domain on a paid plan).
- Models live on **Google Drive** (`MWIZERWA_MODELS_DIR`) and load at startup (LLM) or lazily (ASR/TTS).
- This is a **demo / shared inference** setup, not a permanently hosted SLA.

---

## 2. Configure the API base URL (required)

**Never hardcode the ngrok hostname** in application code in multiple places.

Use one variable everywhere:

```bash
MWIZERWA_API_BASE_URL=https://YOUR-CURRENT-NGROK-SUBDOMAIN.ngrok-free.app
```

- **No trailing slash** on the base URL.
- All requests: `{MWIZERWA_API_BASE_URL}/chat`, `/health`, etc.
- When ngrok restarts, **only update this one value** (env file, secrets manager, or config table).

---

## 3. Discovering the current endpoint (stable pointer)

The ngrok URL is unstable; the **GitHub repo** can hold a small registry file:

**Raw URL (stable):**

`https://raw.githubusercontent.com/Mindora-AI/ai_engine/main/deployment/current_endpoint.json`

**Schema:**

| Field | Meaning |
|-------|---------|
| `base_url` | Active `https://…` ngrok origin (no trailing slash) |
| `status` | e.g. `online`, `offline`, `starting` |
| `last_updated` | ISO-8601 UTC when the operator last published |
| `api_docs_url` | Optional `{base_url}/docs` |

**Important:** Creating `deployment/current_endpoint.json` does **not** auto-update it. After each Colab + ngrok start, the **operator** must:

1. Copy the new ngrok HTTPS URL.
2. Edit `deployment/current_endpoint.json` (or run a small script).
3. Commit and push to `main` (or publish via your org’s approved process).

**Optional future automation (not shipped today):** a Colab cell that writes `base_url` and pushes via a **GitHub token stored in Colab Secrets** — never commit tokens. Until that exists, treat the JSON file as **manually maintained**.

**Client pattern:**

```python
import os, requests

REGISTRY = "https://raw.githubusercontent.com/Mindora-AI/ai_engine/main/deployment/current_endpoint.json"

def resolve_base_url() -> str:
    env = os.environ.get("MWIZERWA_API_BASE_URL")
    if env:
        return env.rstrip("/")
    data = requests.get(REGISTRY, timeout=10).json()
    if data.get("status") != "online" or not data.get("base_url"):
        raise RuntimeError("MWIZERWA endpoint registry reports offline")
    return data["base_url"].rstrip("/")
```

Prefer **env override** in production integrations so you are not blocked on git push latency.

---

## 4. Colab operator quick start

1. Open `notebooks/MWIZERWA_Colab.ipynb`, GPU runtime, run setup cells through model validation.
2. Install API deps: `pip install -r requirements-api.txt`
3. Start server:

   ```bash
   export MWIZERWA_API_HOST=0.0.0.0
   export MWIZERWA_API_PORT=8000
   python scripts/serve_api.py
   ```

4. In another Colab cell or terminal: `ngrok http 8000` (authenticate ngrok in Colab per ngrok docs).
5. Call `GET {ngrok_url}/health` until `engine_initialized: true`.
6. Update `deployment/current_endpoint.json` with the new `base_url` and push (see §3).

---

## 5. API reference

Base path: `{MWIZERWA_API_BASE_URL}`

Interactive docs: `{BASE}/docs` · OpenAPI: `{BASE}/openapi.json`

### Supported languages

| Code | Language | Chat | TTS engine |
|------|----------|------|------------|
| `rw` | Kinyarwanda | Auto-detected per turn | KinyaFlex |
| `en` | English | Auto-detected per turn | Piper |
| `fr` | French | Auto-detected per turn | Piper |

Optional `language` on `/chat` is a hint (`rw` / `en` / `fr`). TTS `/tts` may omit `language` to auto-detect from text.

**Upload limit:** `MWIZERWA_MAX_UPLOAD_BYTES` (default **25 MB**). Extensions: `.wav`, `.webm`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.mpeg`, `.mp4`.

---

### `GET /health`

| | |
|---|---|
| **Purpose** | Check whether the LLM finished loading |
| **Response** | `application/json` |

```json
{"status": "ready", "engine_initialized": true}
```

`status` is `unavailable` when the GGUF failed to load. HTTP **200** even when unavailable (check fields).

---

### `POST /chat`

| | |
|---|---|
| **Content-Type** | `application/json` |
| **Purpose** | Text turn through full MWIZERWA pipeline (safety, exercises, AfriqueQwen) |

**Body**

| Field | Required | Type |
|-------|----------|------|
| `user_text` | yes | string (non-empty after trim) |
| `session_id` | no | string — reuse for same conversation |
| `mode` | no | `gentle` \| `normal` \| `motivational` |
| `language` | no | `rw` \| `en` \| `fr` hint |

**Example request**

```json
{
  "user_text": "Muraho, ndumva mpayijwe n'ubwoba.",
  "session_id": "client-session-42",
  "mode": "normal"
}
```

**Example response (200)**

```json
{
  "session_id": "client-session-42",
  "response_text": "...",
  "mode": "normal",
  "language": "rw",
  "metadata": {
    "interaction_intent": "LISTENING",
    "requires_clarification": false,
    "safety_action": "normal_conversation"
  }
}
```

`metadata` only includes [safe public keys](src/api/service.py) (no file paths or raw safety objects).

**Errors:** `422` validation · `503` engine not ready · `400` bad mode/language · `500` generic failure message

---

### `POST /tts`

| | |
|---|---|
| **Content-Type** | `multipart/form-data` |
| **Purpose** | Synthesize speech from text |

| Field | Required | Type |
|-------|----------|------|
| `text` | yes | string |
| `language` | no | `rw` \| `en` \| `fr` |
| `speaker_id` | no | int 0–2 (KinyaFlex speakers) |

**Response (200):** `audio/wav` bytes (`Content-Disposition: attachment; filename="speech.wav"`).

**Errors:** `400` empty/unsupported language · `503` missing TTS model files · `500` synthesis failed

---

### `POST /transcribe`

| | |
|---|---|
| **Content-Type** | `multipart/form-data` |
| **Purpose** | Speech-to-text (ASR) |

| Field | Required | Type |
|-------|----------|------|
| `audio` | yes | file upload |

**Example response (200)**

```json
{"transcription": "..."}
```

**Errors:** `400` bad file type / empty transcription · `413` file too large · `503` ASR unavailable · `500` transcription failed

---

### `POST /voice`

| | |
|---|---|
| **Content-Type** | `multipart/form-data` |
| **Purpose** | ASR → chat → TTS in one call (same session model as `/chat`) |

| Field | Required | Type |
|-------|----------|------|
| `audio` | yes | file |
| `session_id` | no | string (continue `/chat` session) |
| `mode` | no | `gentle` \| `normal` \| `motivational` |
| `speaker_id` | no | int 0–2 |

**Example response (200)**

```json
{
  "session_id": "…",
  "transcription": "user speech text",
  "response_text": "assistant reply",
  "mode": "normal",
  "language": "rw",
  "metadata": {},
  "audio_base64": "<base64-encoded WAV>",
  "audio_media_type": "audio/wav"
}
```

Decode `audio_base64` to play or save assistant audio.

**Errors:** same families as transcribe + chat (`503` if LLM not ready)

---

## 6. Integration examples

Set once:

```bash
export MWIZERWA_API_BASE_URL=https://YOUR-NGROK-URL.ngrok-free.app
```

### Python (`requests`)

```python
import os, base64, requests

BASE = os.environ["MWIZERWA_API_BASE_URL"].rstrip("/")

# Health
r = requests.get(f"{BASE}/health", timeout=30)
r.raise_for_status()
assert r.json()["engine_initialized"]

# Chat
r = requests.post(
    f"{BASE}/chat",
    json={"user_text": "Hello, I feel anxious."},
    timeout=120,
)
r.raise_for_status()
session_id = r.json()["session_id"]

# TTS → WAV file
r = requests.post(
    f"{BASE}/tts",
    data={"text": "Hello", "language": "en"},
    timeout=120,
)
r.raise_for_status()
open("out.wav", "wb").write(r.content)

# Transcribe
with open("user.wav", "rb") as f:
    r = requests.post(
        f"{BASE}/transcribe",
        files={"audio": ("user.wav", f, "audio/wav")},
        timeout=180,
    )
r.raise_for_status()
print(r.json()["transcription"])

# Voice (reuse session_id from chat)
with open("user.wav", "rb") as f:
    r = requests.post(
        f"{BASE}/voice",
        data={"session_id": session_id},
        files={"audio": ("user.wav", f, "audio/wav")},
        timeout=300,
    )
r.raise_for_status()
body = r.json()
audio = base64.b64decode(body["audio_base64"])
open("assistant.wav", "wb").write(audio)
```

### JavaScript (`fetch`)

```javascript
const BASE = process.env.MWIZERWA_API_BASE_URL.replace(/\/$/, "");

const health = await fetch(`${BASE}/health`);
const h = await health.json();
if (!h.engine_initialized) throw new Error("API not ready");

const chat = await fetch(`${BASE}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ user_text: "Bonjour" }),
});
const { session_id, response_text } = await chat.json();

const form = new FormData();
form.append("text", response_text);
form.append("language", "fr");
const tts = await fetch(`${BASE}/tts`, { method: "POST", body: form });
const wavBytes = await tts.arrayBuffer();
// use wavBytes in AudioContext or download blob
```

**ngrok free tier:** some clients need header `ngrok-skip-browser-warning: true` on requests (check ngrok docs).

---

## 7. Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| Connection refused / timeout | Colab or ngrok down | Operator restarts runtime + ngrok; refresh `MWIZERWA_API_BASE_URL` or registry |
| **404** on all paths | Wrong base URL or old ngrok URL | Fix base URL; fetch registry |
| **422** | Invalid JSON or empty `user_text` | Fix client payload |
| **413** | Audio too large | Shorter clip or raise `MWIZERWA_MAX_UPLOAD_BYTES` on server |
| **503** on `/chat` | LLM not loaded | Wait for `/health`; check GGUF on Drive |
| **503** on `/tts` / `/voice` | Missing KinyaFlex/Piper files | Run `download_models.py` on Colab |
| **500** | Server error | Retry; if persistent, check Colab logs (not exposed in API body) |
| Slow first `/transcribe` or `/voice` | ASR model download to HF cache | Warm up once; cache on Drive via `HF_HOME` |

**Client vs server:** If `GET /health` works from curl but not from your app, suspect local config, TLS, or ngrok browser warning. If health fails everywhere, the hosted stack is down.

---

## 8. Limitations

- Temporary Colab + ngrok hosting; **no uptime SLA**.
- URL rotation is **expected**.
- Cold start and first ASR/TTS load add **latency**.
- ASR is optimized for **Kinyarwanda health-domain** audio; other languages are best-effort.
- Concurrent requests are **serialized** around LLM/voice paths on a single Colab GPU.
- Not a replacement for on-device or dedicated server deployment.

---

## 9. Related docs

- [`docs/API.md`](docs/API.md) — concise endpoint reference
- [`docs/COLAB.md`](docs/COLAB.md) — Colab model setup
- [`scripts/serve_api.py`](scripts/serve_api.py) — server entrypoint