import os
import json
import time
import asyncio
import threading
import queue as sync_queue
import re

import websockets
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from livekit import api as livekit_api
from livekit.protocol import room as livekit_room
from livekit.protocol import agent_dispatch as livekit_agent_dispatch

from app.config import settings
from app.agents import run_response, run_response_stream
from app.tts import generate_tts
from app.asr import transcribe_audio, is_google_asr_configured

app = FastAPI(title="Mario Vargas Llosa Conversational Avatar API")

# Permitir CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ELEVEN_MODEL = "eleven_flash_v2_5"
STREAM_SAMPLE_RATE = 24000
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")

# Definición del esquema del Request de Chat
class ChatRequest(BaseModel):
    message: str
    gemini_key: str = None
    elevenlabs_key: str = None
    voice_id: str = None

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.get("/api/config")
async def get_config():
    """Retorna el estado de las configuraciones de las APIs."""
    return {
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "elevenlabs_configured": bool(settings.ELEVENLABS_API_KEY),
        "google_asr_configured": is_google_asr_configured(),
        "default_voice_id": settings.ELEVENLABS_VOICE_ID,
        "livekit_url": settings.LIVEKIT_URL,
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Endpoint principal: genera la respuesta del avatar (una sola llamada a Gemini)
    y la sintetiza con ElevenLabs (con caché local por hash del texto).
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    start_time = time.time()

    t0 = time.time()
    stylist_output = run_response(request.message, request.gemini_key)
    llm_time = int((time.time() - t0) * 1000)

    t0 = time.time()
    audio_filename, is_cache_hit = await generate_tts(
        stylist_output,
        request.elevenlabs_key,
        request.voice_id
    )
    tts_time = int((time.time() - t0) * 1000)

    total_time = int((time.time() - start_time) * 1000)

    audio_url = f"/cache/{audio_filename}" if audio_filename else None

    return {
        "stylist": stylist_output,
        "audio_url": audio_url,
        "latencies": {
            "llm_ms": llm_time,
            "tts_ms": tts_time,
            "total_ms": total_time
        },
        "cache_hit": is_cache_hit
    }

async def _consume_stream(gen):
    """Puentea el generador síncrono de Gemini (google-generativeai) hacia asyncio,
    corriéndolo en un hilo aparte y entregando los fragmentos vía una cola."""
    out_queue: sync_queue.Queue = sync_queue.Queue()
    SENTINEL = object()

    def worker():
        try:
            for piece in gen:
                out_queue.put(piece)
        finally:
            out_queue.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        piece = await asyncio.to_thread(out_queue.get)
        if piece is SENTINEL:
            break
        yield piece

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat de texto puro: transmite la respuesta del avatar por SSE, palabra por palabra
    a medida que el LLM la genera. Sin síntesis de voz — lo usa el chat de texto del
    frontend, que no reproduce audio (la llamada de voz vive aparte, vía LiveKit).
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    async def event_generator():
        try:
            yield sse("status", {"text": "escribiendo..."})

            full_text = ""
            async for piece in _consume_stream(run_response_stream(request.message, request.gemini_key)):
                full_text += piece
                yield sse("token", {"text": piece})

            full_text = full_text.strip()

            yield sse("done", {"full_text": full_text})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat/stream-v2")
async def chat_stream_v2(request: ChatRequest):
    """
    Streaming avanzado: transmite el texto por SSE a medida que Gemini lo genera y,
    en paralelo, envía cada oración completa al WebSocket de streaming de ElevenLabs,
    reenviando los chunks de audio (PCM base64) al cliente apenas llegan.
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    eleven_key = request.elevenlabs_key or settings.ELEVENLABS_API_KEY
    voice_id = request.voice_id or settings.ELEVENLABS_VOICE_ID

    async def event_generator():
        audio_queue: asyncio.Queue = asyncio.Queue()
        ws = None
        ws_ready = False

        async def ws_receiver():
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if msg.get("audio"):
                        await audio_queue.put({
                            "audio": msg["audio"],
                            "sample_rate": STREAM_SAMPLE_RATE,
                            "is_final": bool(msg.get("isFinal")),
                        })
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[ElevenLabs WS] Conexión cerrada: code={e.code} reason={e.reason!r}")

        receiver_task = None
        if eleven_key and voice_id:
            ws_url = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
                f"?model_id={ELEVEN_MODEL}&output_format=pcm_{STREAM_SAMPLE_RATE}"
            )
            try:
                ws = await websockets.connect(ws_url)
                await ws.send(json.dumps({
                    "text": " ",
                    "voice_settings": {"stability": 0.55, "similarity_boost": 0.8},
                    "xi_api_key": eleven_key,
                    "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
                }))
                ws_ready = True
                receiver_task = asyncio.create_task(ws_receiver())
                print("[ElevenLabs WS] Conexión abierta")
            except Exception as e:
                print(f"[ElevenLabs WS] Error al conectar: {e}")
                ws = None

        async def send_sentence(sentence: str):
            sentence = sentence.strip()
            if ws_ready and ws and sentence:
                try:
                    print(f"[ElevenLabs WS] Enviando oración ({len(sentence)} chars): {sentence[:50]!r}...")
                    await ws.send(json.dumps({"text": sentence + " "}))
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[ElevenLabs WS] send() sobre conexión cerrada: code={e.code} reason={e.reason!r}")

        def drain_audio_queue():
            while not audio_queue.empty():
                yield audio_queue.get_nowait()

        try:
            yield sse("status", {"text": "escribiendo..."})

            full_text = ""
            sentence_buffer = ""

            async for piece in _consume_stream(run_response_stream(request.message, request.gemini_key)):
                full_text += piece
                yield sse("token", {"text": piece})

                sentence_buffer += piece
                sentences = SENTENCE_SPLIT.split(sentence_buffer)
                if len(sentences) > 1:
                    for complete in sentences[:-1]:
                        await send_sentence(complete)
                    sentence_buffer = sentences[-1]

                for chunk in drain_audio_queue():
                    yield sse("audio_chunk", chunk)

            if sentence_buffer.strip():
                await send_sentence(sentence_buffer)

            full_text = full_text.strip()

            if ws_ready and ws:
                try:
                    await ws.send(json.dumps({"text": ""}))
                except websockets.exceptions.ConnectionClosed:
                    pass
                if receiver_task:
                    try:
                        await asyncio.wait_for(receiver_task, timeout=3.0)
                    except asyncio.TimeoutError:
                        pass
                for chunk in drain_audio_queue():
                    yield sse("audio_chunk", chunk)
                try:
                    await ws.close()
                except Exception:
                    pass

            yield sse("done", {"full_text": full_text, "sample_rate": STREAM_SAMPLE_RATE})
        except Exception as e:
            yield sse("error", {"message": str(e)})
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/livekit/token")
async def livekit_token(room: str = Query("mvll-room"), participant: str = Query("visitante")):
    """Emite un token de acceso a LiveKit Cloud para que el frontend se una a una
    sala de voz atendida por el agente 'mvll_local'."""
    if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="Credenciales de LiveKit no configuradas en el backend.")

    # El campo "agents" de RoomConfiguration es lo que realmente le indica a LiveKit
    # que despache la sala al worker "mvll_local" en cuanto se cree — a diferencia de
    # with_metadata(), que solo pone un dato decorativo en el participante y no dispara
    # ningún despacho por sí solo (esa era la causa de que la llamada nunca contestara).
    token = (
        livekit_api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(participant)
        .with_grants(livekit_api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
        ))
        .with_room_config(livekit_room.RoomConfiguration(
            agents=[livekit_agent_dispatch.RoomAgentDispatch(agent_name="mvll_local")],
        ))
    )
    return {"token": token.to_jwt()}

@app.post("/api/asr")
async def asr(file: UploadFile = File(...)):
    """
    Endpoint de ASR: Recibe un archivo de audio y lo transcribe usando
    Google Cloud Speech-to-Text.
    """
    audio_bytes = await file.read()
    transcript = await transcribe_audio(audio_bytes, file.content_type)

    return {
        "transcript": transcript,
        "success": bool(transcript)
    }

# 4. Servir archivos estáticos del Caché de Audio
app.mount("/cache", StaticFiles(directory=settings.CACHE_DIR), name="cache")

# 5. Servir archivos del Frontend
# Localizar la ruta del frontend (c:\Users\user\Desktop\MVLL\frontend)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(os.path.dirname(base_dir), "frontend")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Advertencia: Directorio de frontend no encontrado en {frontend_path}")
