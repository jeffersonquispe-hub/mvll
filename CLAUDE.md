# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An MVP conversational avatar of Mario Vargas Llosa (Nobel laureate writer). User text/voice input is sent to
an LLM in a **single fused call** per turn: the model is instructed to silently reason about an intellectual
stance, 2-3 supporting arguments, and literary references before writing — but to output only the final
answer, already in Vargas Llosa's literary prose style — and the result is spoken aloud via an ElevenLabs
cloned voice (Voice ID `7B1CbnTtwwTp1CCGjRzn`). All prompts, comments, and UI copy are in Spanish — keep new
agent/user-facing text in Spanish to match.

**LLM/STT provider order is currently AWS-first, Google-second.** Gemini's free tier has a **20
requests/day** cap (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) that gets exhausted quickly during
normal development, so both the HTTP backend and the voice agent try **Claude via AWS Bedrock first** and
fall back to Gemini only if Bedrock isn't configured or fails. Same pattern for ASR: Google Cloud
Speech-to-Text first, AWS Transcribe as fallback (this one Google-first since Transcribe is slower — see TTS
caching / ASR fallback sections below). If Gemini's daily quota resets and you want it primary again, or if
Bedrock model access changes, the call order is swapped back easily — see `run_response()`/
`run_response_stream()` in `agents.py` and the `FallbackAdapter` list in `mvll_agent.py`.

This single-call design replaced an earlier two-stage "Director" (plans the stance/arguments as JSON) +
"Estilista" (rewrites the plan as prose) pipeline that made two sequential Gemini calls per turn. The
two-step scaffolding was collapsed into one prompt to halve LLM round-trip latency; nothing in the frontend
ever consumed the intermediate Director JSON, so it was dropped entirely rather than kept for compatibility.

There is no git repository initialized in this directory.

## Repo layout — one HTTP backend (Python) + one LiveKit voice agent

There used to be two interchangeable HTTP backends (Node and Python). The Node one (`backend_node/`) was
removed by explicit choice — **`backend/app/` (FastAPI/Python) is the only HTTP backend** and is the source
of truth for the chat pipeline:

- `backend/app/agents.py` — the fused prompt (`build_fused_prompt`), `run_response()` (non-streaming) and
  `run_response_stream()` (generator used by the SSE endpoints). Tries `bedrock_fallback.py` (Claude via AWS
  Bedrock) first, then Gemini.
- `backend/app/bedrock_fallback.py` — Claude via AWS Bedrock (`boto3`, not the official `anthropic` SDK — see
  note below). Model is `us.anthropic.claude-sonnet-4-6`; this AWS account doesn't have model access approved
  for Opus 4.8/Sonnet 5/Fable 5 (`AccessDeniedException`) — Sonnet 4.6 and Haiku 4.5 are the only ones that
  work. Check access again in the Bedrock console before assuming a newer model is unavailable forever.
- `backend/app/tts.py` — ElevenLabs synthesis with SHA256-hash disk caching. No AWS fallback (would need
  Amazon Polly) — ElevenLabs hasn't failed in practice so this wasn't built.
- `backend/app/asr.py` — Google Cloud Speech-to-Text first, falls back to `transcribe_fallback.py` (AWS
  Transcribe) if unconfigured/fails. Not called by the current frontend (prefers browser-native
  `webkitSpeechRecognition`) — this path exists but is effectively dead code in the shipped UI today.
- `backend/app/transcribe_fallback.py` — AWS Transcribe fallback. Unlike Bedrock, Transcribe has **no
  synchronous API** — this uploads the audio to the S3 bucket `mvll-asr-fallback-183150676819` (public access
  blocked, 1-day lifecycle expiry, dedicated to this project), starts a transcription job, polls for
  completion (~10-15s for short clips), reads the result, then deletes both the audio and result objects.
- `backend/app/config.py` — loads the root `.env` (resolved relative to `backend/`).
- `backend/app/main.py` — routes: `/api/config`, `/api/chat`, `/api/chat/stream`, `/api/chat/stream-v2`,
  `/api/livekit/token`, `/api/asr`, plus static mounts for `/cache` and the frontend.

**The official `anthropic` Python SDK cannot be used on this machine** — its `jiter` native dependency is
blocked by a Windows Application Control policy (`ImportError: DLL load failed ... blocked this file`). That's
why the Bedrock integration is hand-rolled with `boto3`'s `bedrock-runtime` client instead of
`AnthropicBedrockMantle`. If setting up on a different machine where the SDK loads fine, switching to the
official client is preferable — but don't assume the blocker is fixed without testing `import anthropic`
first.

`agent/mvll_agent.py` is the **single canonical LiveKit voice agent** (STT→LLM→TTS over WebRTC, for
real-time phone-call-style conversation instead of request/response HTTP). It uses the official
`livekit-agents[google,elevenlabs,silero]` plugins, plus `livekit-plugins-aws` for the Bedrock LLM fallback.
The `llm=` passed to `AgentSession` is a `livekit.agents.llm.FallbackAdapter([aws_llm.LLM(...), google.LLM(...)])`
— AWS first, Gemini second, same order/reasoning as the HTTP backend. The MVLL persona is
`MVLL_SYSTEM_PROMPT`, a **separate, hand-written duplicate** of `agents.py`'s `build_fused_prompt()` — the two
are not shared code (see "Duplicated persona prompt" below). It's a separate always-on process, not spawned
by the HTTP backend; the backend's only connection to it is minting the join token at `/api/livekit/token`,
which the frontend uses to connect directly to LiveKit Cloud. That token must include
`RoomConfiguration(agents=[RoomAgentDispatch(agent_name="mvll_local")])` (via `.with_room_config()`) for
LiveKit to actually dispatch the room to this worker — setting `agent_name` in the token's `metadata` field
alone does **nothing** for dispatch (that was a bug here for a while: the room would connect fine but no
agent would ever join).

`frontend/` is static HTML/CSS/JS served by the Python backend at `/`. It has no build step and talks to the
backend on the same origin via relative `fetch()` calls.

**Duplicated persona prompt.** `backend/app/agents.py`'s `build_fused_prompt()` and `agent/mvll_agent.py`'s
`MVLL_SYSTEM_PROMPT` define the same Vargas Llosa persona independently, because the two run under
incompatible execution models (stateless HTTP request/response vs. a long-lived LiveKit `AgentSession`) and
don't share a code path. When asked to change agent behavior (prompt content, tone, output format), update
**both** files — nothing keeps them in sync automatically.

**Convenience launchers** at the repo root — `start-backend.cmd` and `start-agent.cmd` — `cd` to the right
directory and invoke the venv's `python.exe` directly, so they work regardless of the caller's current
directory or whether a global `python` is on PATH (this machine has no global Python — bare `python` resolves
to the Windows Store app-execution alias). `start-agent.cmd` also auto-restarts the agent process if it
crashes. Prefer these over manually `cd`-ing and activating venvs when starting either service.

## Running

First-time setup:
```powershell
cd backend
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```
After that, use `start-backend.cmd` at the repo root to start it (see "Convenience launchers" above) — or
`.\.venv\Scripts\python.exe run.py` from `backend/` directly. Serves the frontend and API at
`http://localhost:<PORT>` (`PORT` from the root `.env`, default `8000`).

**LiveKit real-time voice agent** (separate process, requires a LiveKit Cloud room; the HTTP backend's
`/api/livekit/token` endpoint mints the join token). First-time setup:
```powershell
cd agent
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```
After that, use `start-agent.cmd` at the repo root.

There are no automated test suites, lint configs, or CI in this repo — `samples/` has two short MP3s (one
clearly in Spanish) used for manually exercising the ASR/ASR-fallback path; otherwise verify changes by
running the backend and exercising the frontend at localhost, or by hitting endpoints directly (e.g.
`POST /api/chat`).

## Configuration

A single root `.env` (`MVLL/.env`) is the source of truth, loaded both by `agent/mvll_agent.py` (absolute
path) and by `backend/app/config.py` (resolved relative to `backend/`, one level up). There used to be a
second, separate `backend/.env` with the same keys duplicated — it was removed since nothing needed two
copies; don't reintroduce a backend-local `.env`.

Keys used: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `GOOGLE_APPLICATION_CREDENTIALS` (a
Google Cloud credentials JSON — used for optional server-side ASR in the HTTP backend, and for
Speech-to-Text in the voice agent), `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `PORT`,
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, and for the AWS fallbacks: `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_S3_ASR_BUCKET`.

If `GEMINI_API_KEY`/credentials are absent, the backend falls back to a hardcoded Spanish mock response
instead of failing — this "Mock Mode" lets the full UI/UX be exercised without any API keys. Preserve this
fallback behavior when touching the LLM call sites.

Note: `GOOGLE_APPLICATION_CREDENTIALS` here points to an `authorized_user`-type ADC credential (from
`gcloud auth application-default login`), not a service account key — it needs a `quota_project_id` field
set (either via `gcloud auth application-default set-quota-project <project>` or by editing the JSON
directly) or Google Cloud Speech-to-Text calls fail with `SERVICE_DISABLED`/quota-project errors even though
the credential itself is valid.

## TTS caching

`backend/app/tts.py` hashes the model's final output text with SHA256 and caches the resulting ElevenLabs
MP3 on disk as `audio_<hash>.mp3` under `settings.CACHE_DIR` (`backend/public/cache`), served statically at
`/cache/...`. Identical response text is never re-synthesized. Keep this hash-on-final-text caching scheme in
mind when changing how/where the model's output is post-processed — any transformation applied after hashing
but before TTS breaks the cache key. The streaming SSE endpoints don't use this cache the same way — see API
shape below.

## HTTP API shape (`backend/app/main.py`)

- `GET /api/config` — reports which credentials are configured, incl. `livekit_url` (used by the frontend to
  decide whether to show the call button and whether to fall back to Mock Mode).
- `POST /api/chat` — fused LLM call → TTS pipeline, returns JSON with `stylist` (final prose), `audio_url`,
  and `latencies: { llm_ms, tts_ms, total_ms }`.
- `POST /api/chat/stream` — same pipeline but SSE, streaming the fused-call tokens (`event: token`) as they
  generate via `run_response_stream()`, then a final MP3 URL (via the same TTS cache) once the full text is
  ready.
- `POST /api/chat/stream-v2` — **the endpoint the frontend actually uses** (`frontend/app.js`). Streams
  tokens from the fused call *and* pipes completed sentences to ElevenLabs' streaming WebSocket API in
  parallel over `websockets`, forwarding raw PCM (24kHz) `audio_chunk` SSE events to the client as they
  arrive, rather than waiting for the whole utterance before synthesizing. The ElevenLabs WebSocket is opened
  immediately on request start. Because `google-generativeai`'s streaming call is synchronous, it runs in a
  background thread and is bridged into the async SSE generator via a queue (`_consume_stream` in
  `main.py`) — keep that bridge if the Gemini SDK call changes.
- `POST /api/asr` — optional server-side Google Cloud STT for audio blobs (multipart `UploadFile`). Not
  called by the current frontend — it prefers the browser's native `webkitSpeechRecognition`.
- `GET /api/livekit/token?room=&participant=` — mints a LiveKit `AccessToken` (via `livekit-api`) for the
  frontend to join a WebRTC room routed to the `mvll_local` voice agent.

## Frontend

Plain HTML/CSS/JS, no build tooling or framework (`frontend/index.html`, `app.js`, `style.css`). Talks to the
backend on the same origin. Falls back to the browser's native `window.speechSynthesis`/
`webkitSpeechRecognition` when backend credentials aren't configured (Mock Mode), and drives an
audio-reactive lip-sync animation off the streamed PCM audio chunks / WebRTC track state.
