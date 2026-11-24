# MWIZERWA — Memory and context summarization (Stage 1)

## Overview

| Capability | Scope |
|------------|--------|
| **Rolling conversation summary** | Per chat session; compresses older turns when token budget requires |
| **Persistent memory** | Per trusted `user_id`; requires explicit consent before storage |

The AfriqueQwen model does **not** provide built-in persistent memory. All summarization is explicit application logic.

## Rolling summary

- Enabled by default (`MWIZERWA_ROLLING_SUMMARY=1`).
- Keeps the last `MWIZERWA_VERBATIM_TAIL_TURNS` turn pairs verbatim (default `2`).
- When the prompt exceeds the budget, oldest pairs are merged via an auxiliary LLM call into `ConversationManager.rolling_summary`.
- Injected ephemerally as `[Conversation Summary]` before recent turns.
- On summarization failure, falls back to trim-only without losing the session.
- Committed only after a successful assistant turn.

## Persistent memory

### AI responsibilities

1. **Extract** structured `memory_proposals` after normal (non–Gate-1) turns when `user_id` is provided.
2. **Never persist** without an explicit accepted consent decision.
3. **Merge** approved items into a compact summary via auxiliary LLM call.

### Application responsibilities (production)

- Provide a **trusted** `user_id` on server-to-server `/chat` calls (not raw client input).
- Present consent UI for each `metadata.memory_proposals[]` entry.
- Record `accepted` / `declined` and call your backend (or dev `POST /memory/consent`).

### Mock / development components

| Component | Replace with |
|-----------|----------------|
| `InMemoryMemoryRepository` | Database adapter implementing `MemoryRepository` |
| `MockConsentGateway` | Application consent service implementing `ConsentGateway` |
| `POST /memory/consent` | Authenticated internal API |

## API

### `POST /chat` (extended, backward compatible)

Optional `user_id` — when set, returns `metadata.memory_proposals` when the extractor produces candidates.

### `POST /memory/consent` (development)

```json
{
  "user_id": "trusted-user-id",
  "proposal_id": "uuid-from-metadata",
  "decision": "accepted"
}
```

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `MWIZERWA_ROLLING_SUMMARY` | `1` | Enable rolling session summarization |
| `MWIZERWA_VERBATIM_TAIL_TURNS` | `2` | Recent turn pairs kept verbatim |