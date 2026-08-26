"""
Agente de voz LiveKit para el Avatar Conversacional de Mario Vargas Llosa.

Pipeline: STT (ElevenLabs Scribe v2 Realtime) → LLM (Gemini 2.5 Flash Lite) → TTS (ElevenLabs)

El agente se conecta a LiveKit Cloud y atiende llamadas WebRTC.
Cada oración generada por Gemini se sintetiza y transmite inmediatamente.

Ejecución:
    cd agent
    uv venv
    .venv\\Scripts\\activate
    uv pip install -r requirements.txt
    python mvll_agent.py dev
"""

import os
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env de forma absoluta y configurar credenciales de Google
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    # Configurar las variables del sistema para LiveKit y ElevenLabs
    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    # GOOGLE_APPLICATION_CREDENTIALS sigue haciendo falta para el LLM (Gemini vía
    # Vertex AI, fallback de Bedrock) — se autentica por ADC leyendo esa variable
    # de entorno directamente, sin que el código la pase explícitamente. El STT ya
    # no usa Google (ver abajo), así que no necesita su propia referencia a la ruta.

    if eleven_key:
        os.environ["ELEVEN_API_KEY"] = eleven_key
        print(f"[Agent Setup] Configured ElevenLabs API Key")

from livekit.agents import (
    AgentSession,
    Agent,
    APITimeoutError,
    DEFAULT_API_CONNECT_OPTIONS,
    JobContext,
    NOT_GIVEN,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.agents import llm
from livekit.agents.llm import LLM, LLMStream, FallbackAdapter
from livekit.plugins import aws as aws_llm
from livekit.plugins import google, elevenlabs, silero
import re

# Modelo Claude vía AWS Bedrock para el agente de voz. Haiku 4.5, no Sonnet: en una
# llamada en tiempo real la latencia importa más que la profundidad del modelo, y las
# respuestas ya están acotadas a 2-4 oraciones cortas por el prompt — Haiku responde
# notablemente más rápido (menor tiempo al primer token y generación) con pérdida de
# calidad mínima para ese tipo de intercambio. El backend HTTP de texto (agents.py)
# sigue usando Sonnet 4.6, donde las respuestas son más largas y la latencia importa menos.
BEDROCK_VOICE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Cuánto esperar entre fragmentos consecutivos de un stream de LLM antes de darlo
# por colgado (ver TimeoutGuardLLM). FallbackAdapter.attempt_timeout (5s) solo cubre
# conectar + recibir los headers — una vez que el stream arrancó, aiohttp deja de
# vigilarlo, así que un stall a mitad de generación queda sin detectar y sin fallback.
# Ajustable sin tocar código: LLM_CHUNK_STALL_TIMEOUT_S en el .env.
LLM_CHUNK_STALL_TIMEOUT_S = float(os.getenv("LLM_CHUNK_STALL_TIMEOUT_S", "8.0"))

# Registro de latencia por turno (JSONL, uno por línea), para evaluación automatizada
# de latencia end-to-end sin tener que leer el log en vivo. Se nutre de
# ChatMessage.metrics, que AgentSession ya calcula por turno (transcription_delay,
# end_of_turn_delay, llm_node_ttft, tts_node_ttfb, e2e_latency, playback_latency) —
# ver el handler de "conversation_item_added" en entrypoint(). Analizar con
# agent/analyze_latency.py.
LATENCY_LOG_PATH = Path(__file__).resolve().parent / "latency_log.jsonl"


class _TimeoutGuardLLMStream(LLMStream):
    """LLMStream que envuelve el stream de otro LLM y le pone un límite de tiempo a
    CADA fragmento (no solo a la conexión inicial). Si no llega un fragmento nuevo
    dentro del plazo, cancela la tarea interna trabada y lanza APITimeoutError —
    la excepción que FallbackAdapter ya sabe capturar para pasar al siguiente LLM."""

    def __init__(
        self,
        guard: "TimeoutGuardLLM",
        *,
        chat_ctx,
        tools,
        conn_options,
        parallel_tool_calls,
        tool_choice,
        extra_kwargs,
    ) -> None:
        super().__init__(guard, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._wrapped_llm = guard._wrapped
        self._chunk_timeout = guard._chunk_timeout
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        inner = self._wrapped_llm.chat(
            chat_ctx=self._chat_ctx,
            tools=self._tools,
            conn_options=self._conn_options,
            parallel_tool_calls=self._parallel_tool_calls,
            tool_choice=self._tool_choice,
            extra_kwargs=self._extra_kwargs,
        )
        async with inner:
            ait = inner.__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(ait.__anext__(), timeout=self._chunk_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise APITimeoutError(
                        f"{self._wrapped_llm.label} stalled mid-stream "
                        f"(no chunk for {self._chunk_timeout}s)"
                    ) from None
                self._event_ch.send_nowait(chunk)


class TimeoutGuardLLM(LLM):
    """Envuelve un LLM cualquiera para que un stall a mitad de stream (que hoy no
    detecta nada en la pila: ni aiohttp, ni el plugin, ni FallbackAdapter) se
    convierta en un error que sí dispare el fallback al siguiente LLM de la lista."""

    def __init__(self, wrapped: LLM, *, chunk_timeout: float) -> None:
        super().__init__()
        self._wrapped = wrapped
        self._chunk_timeout = chunk_timeout

    @property
    def label(self) -> str:
        return f"TimeoutGuardLLM({self._wrapped.label})"

    @property
    def model(self) -> str:
        return self._wrapped.model

    @property
    def provider(self) -> str:
        return self._wrapped.provider

    def prewarm(self, *args, **kwargs) -> None:
        self._wrapped.prewarm(*args, **kwargs)

    async def aclose(self) -> None:
        await self._wrapped.aclose()

    def chat(
        self,
        *,
        chat_ctx,
        tools=None,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> LLMStream:
        return _TimeoutGuardLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )


# ============================================================
# SYSTEM PROMPT — Personalidad de Mario Vargas Llosa
# ============================================================
MVLL_SYSTEM_PROMPT = """Eres Mario Vargas Llosa, el célebre escritor peruano-español, Premio Nobel de Literatura 2010, respondiendo en una conversación de voz en tiempo real.

## Tu identidad
- Eres un intelectual de convicciones liberales clásicas profundas
- Crees en la libertad individual, la democracia, la separación de poderes
- Criticas severamente los autoritarismos (de izquierda y derecha)
- Estás obsesionado con el poder de la ficción y la creación literaria como rebelión contra la realidad
- Tus obras incluyen: 'Conversación en La Catedral', 'La fiesta del Chivo', 'La ciudad y los perros', 'La guerra del fin del mundo'

## Tu estilo de habla
- Habla SIEMPRE en primera persona: 'Yo', 'A lo largo de mi vida...', 'Siempre he sostenido...'
- Usa vocabulario intelectual: 'ficción', 'artificio', 'desvarío', 'fanatismo', 'disciplina implacable'
- Frases cadenciosas con cláusulas coordinadas por comas y puntos y coma
- NUNCA uses saludos genéricos como 'Hola' o 'Buenos días'
- Transmite pasión por el oficio de escribir como acto de rebelión

## Cómo razonar antes de responder
Antes de hablar, piensa en silencio (sin decirlo en voz alta) cuál es tu postura ante lo que te preguntan,
1 o 2 argumentos conceptuales que la sustenten, y si aplica, una referencia concreta a tus novelas, ensayos
u otros pensadores/escritores afines (Flaubert, Sartre, Popper). Luego responde directamente incorporando
esas ideas de forma natural, sin enumerarlas ni exponer el razonamiento — es una conversación hablada, no
una exposición.

REGLA ABSOLUTA: todo lo que escribas se sintetiza y se reproduce como audio, sin ningún filtro — tu
respuesta debe ser ÚNICAMENTE las palabras que dirías en voz alta, empezando directo con ellas. Nunca
escribas etiquetas, encabezados ni marcadores de tu proceso interno — cosas como "(Silencio mental)",
"Postura:", "Argumento:", "Referencia:" o "(Respuesta)" — ni siquiera para separar tu razonamiento de la
respuesta final. Si alguna vez sentís el impulso de escribir algo así, es una señal de que tenés que borrarlo
y quedarte solo con la respuesta.

## Reglas para conversación de voz
- Responde de forma CONCISA: 2-4 oraciones máximo (es una conversación hablada, no un ensayo)
- Sé directo y ve al grano
- Adapta la extensión al tipo de pregunta: preguntas simples → respuestas cortas
- Habla siempre en español
"""


# Gemini 2.5 Flash Lite ignora la instrucción del prompt de razonar "en silencio" con
# más frecuencia de la esperada, y escribe el andamiaje de razonamiento como texto
# normal (ej. "(Silencio mental)\nPostura: ...\nArgumento: ...\n\n(Respuesta)\n..."),
# que sin este filtro se sintetiza y se escucha en voz alta tal cual. El prompt ya lo
# prohíbe explícitamente (ver MVLL_SYSTEM_PROMPT), pero eso solo reduce la frecuencia;
# esto es la garantía a nivel de código de que nunca llega a hablarse.
_LEAK_LABEL = r"(?:Silencio mental|Postura|Argumento|Referencia|Respuesta)(?:\s*\d+)?"
_MARKUP = r"[\s\*\-_]*"  # tolera bullets/negrita markdown alrededor de la etiqueta
_REASONING_LEAK_RE = re.compile(
    rf"^{_MARKUP}\(?{_MARKUP}{_LEAK_LABEL}{_MARKUP}\)?{_MARKUP}:"
    rf"|^{_MARKUP}\({_MARKUP}{_LEAK_LABEL}{_MARKUP}\){_MARKUP}$",
    re.IGNORECASE,
)

# Si una respuesta arranca con "(" pero no aparece una línea en blanco dentro de
# este umbral de caracteres, se asume que no era un preámbulo de razonamiento (ver
# MVLLAgent.llm_node) y se deja de intentar filtrarla.
_PREAMBLE_SAFETY_CAP = 400


class MVLLAgent(Agent):
    """Agente conversacional que encarna a Mario Vargas Llosa."""

    def __init__(self):
        super().__init__(
            instructions=MVLL_SYSTEM_PROMPT,
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Filtra el razonamiento que el LLM a veces escribe como texto plano pese a
        la instrucción del prompt de pensarlo "en silencio". Dos capas, porque el
        modelo inventa etiquetas nuevas cada vez ("Postura:", "(Piensa: ...)", etc.)
        y perseguir palabras puntuales es un juego sin fin:
        1) Si la respuesta entera arranca con "(", es un preámbulo de razonamiento
           en TODOS los casos vistos hasta ahora — se descarta todo hasta la primera
           línea en blanco, sea cual sea la etiqueta que use.
        2) Además, cualquier línea suelta con una etiqueta conocida se filtra por
           separado (ver _REASONING_LEAK_RE), por si aparece sin el paréntesis.
        """
        buffer = ""
        # None = todavía no se vio el primer carácter no blanco de la respuesta.
        stripping_preamble: bool | None = None
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                content = chunk.delta.content
            elif isinstance(chunk, str):
                content = chunk
            else:
                yield chunk
                continue

            buffer += content

            if stripping_preamble is None:
                stripped = buffer.lstrip()
                if not stripped:
                    continue
                stripping_preamble = stripped.startswith("(")

            if stripping_preamble:
                idx = buffer.find("\n\n")
                if idx != -1:
                    buffer = buffer[idx + 2 :]
                    stripping_preamble = False
                elif len(buffer) < _PREAMBLE_SAFETY_CAP:
                    # Todavía esperando el final del preámbulo (la línea en blanco).
                    continue
                else:
                    # Pasado el umbral sin encontrar línea en blanco: probablemente
                    # nunca fue un preámbulo de razonamiento. Dejar de filtrar.
                    stripping_preamble = False

            if "\n" not in buffer:
                continue
            *complete_lines, buffer = buffer.split("\n")
            cleaned = "\n".join(
                line for line in complete_lines if not _REASONING_LEAK_RE.match(line)
            )
            if not cleaned:
                continue
            if isinstance(chunk, llm.ChatChunk):
                yield llm.ChatChunk(
                    id=chunk.id,
                    delta=llm.ChoiceDelta(role=chunk.delta.role, content=cleaned + "\n"),
                )
            else:
                yield cleaned + "\n"

        if stripping_preamble:
            # La respuesta entera terminó sin nunca encontrar la línea en blanco —
            # probablemente no era un preámbulo, así que se deja pasar tal cual en
            # vez de arriesgarse a dejar a Mario en silencio.
            idx = buffer.find("\n\n")
            if idx != -1:
                buffer = buffer[idx + 2 :]
        if buffer and not _REASONING_LEAK_RE.match(buffer):
            yield buffer


def prewarm(proc) -> None:
    """Carga el modelo VAD una sola vez por proceso worker (no por llamada). Antes
    `silero.VAD.load()` se llamaba dentro de entrypoint() y recargaba el modelo ONNX
    desde disco en cada llamada, sumando a los ~2.5s de arranque antes de que Mario
    diga la primera palabra del saludo. Con prewarm_fnc, el modelo ya está cargado y
    esperando en el proceso idle cuando llega el job."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Punto de entrada del agente — se ejecuta cuando un participante entra a la sala."""

    await ctx.connect()

    session = AgentSession(
        # Detección de actividad de voz — precargada en prewarm(), no acá.
        vad=ctx.proc.userdata["vad"],

        # Endpointing más agresivo que el default (min_delay=0.3s/max_delay=2.5s para el
        # detector semántico streaming) — analyze_latency.py mostró end_of_turn_delay como
        # el mayor componente de la latencia end-to-end. Baja el margen de espera tras
        # detectar silencio; trade-off: una pausa larga a mitad de una frase se puede
        # cortar antes de tiempo.
        #
        # resume_false_interruption=False: el default (True) hace que, si el detector
        # adaptativo de interrupciones clasifica tu barge-in como "falso" (backchannel/
        # ruido), Mario retome y TERMINE la respuesta vieja antes de atender lo que
        # preguntaste — se vio en una prueba real ("resumed false interrupted speech" en
        # el log, dos veces seguidas) y se sentía como si respondiera preguntas
        # anteriores. Con False, cualquier barge-in corta la respuesta en curso para
        # siempre, aunque haya sido ruido — mejor trade-off para una demo en vivo.
        turn_handling={
            "endpointing": {"min_delay": 0.3, "max_delay": 1.1},
            "interruption": {"resume_false_interruption": False},
        },

        # Speech-to-Text en español — ElevenLabs Scribe v2 Realtime (streaming por
        # WebSocket, <150ms de latencia). Reemplaza a Google Cloud STT: no requiere
        # credenciales aparte, usa la misma ELEVEN_API_KEY que el TTS de abajo.
        #
        # server_vad es obligatorio para que finalice rápido: sin esto, el plugin usa
        # commit_strategy="manual" (depende de que LiveKit le mande un flush explícito
        # a ElevenLabs) y en la práctica se vio el transcript parcial repetirse ~22
        # veces sin finalizar durante más de 20s (analyze_latency.py). Con server_vad,
        # es el propio servidor de ElevenLabs el que detecta el silencio y corta
        # (commit_strategy="vad") — el modo pensado para este modelo.
        stt=elevenlabs.STT(
            model="scribe_v2_realtime",
            language_code="es",
            server_vad={
                "vad_silence_threshold_secs": 0.5,
                "min_silence_duration_ms": 500,
            },
        ),

        # LLM: Gemini vía Vertex AI primero, con Claude (AWS Bedrock) como respaldo — orden
        # invertido respecto al backend HTTP (agents.py) mientras dure esto: la cuenta de AWS
        # nueva (435788423620, usuario "voice") todavía no tiene aprobado el acceso a modelos
        # Bedrock ("Access to Bedrock models is not allowed for this account"), así que
        # probarlo primero solo suma ~0.5s de intento fallido a cada turno, medido con
        # analyze_latency.py. Volver a AWS-primero apenas se apruebe el acceso en la consola.
        #
        # Ambos LLM van envueltos en TimeoutGuardLLM: FallbackAdapter.attempt_timeout solo cubre
        # conectar + recibir headers (vía aiohttp ClientTimeout(total=...), que deja de vigilar
        # apenas arranca el stream) — si el stream se traba a mitad de generación, hoy nada lo
        # detecta ni dispara el fallback (causó un cuelgue real de 37s en una llamada). El wrapper
        # le pone un límite a cada fragmento del stream, no solo a la conexión inicial.
        llm=FallbackAdapter([
            TimeoutGuardLLM(
                google.LLM(
                    model="gemini-2.5-flash-lite",
                    vertexai=True,
                    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                    location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
                    temperature=0.7,
                ),
                chunk_timeout=LLM_CHUNK_STALL_TIMEOUT_S,
            ),
            TimeoutGuardLLM(
                aws_llm.LLM(
                    model=BEDROCK_VOICE_MODEL,
                    api_key=os.getenv("AWS_ACCESS_KEY_ID"),
                    api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                ),
                chunk_timeout=LLM_CHUNK_STALL_TIMEOUT_S,
            ),
        ]),

        # Text-to-Speech: ElevenLabs con la voz clonada de MVLL
        tts=elevenlabs.TTS(
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "7B1CbnTtwwTp1CCGjRzn"),
            model="eleven_flash_v2_5",
            language="es",
        ),
    )

    # Latencia por turno: cada ChatMessage trae su propio MetricsReport (dict con
    # claves opcionales según el rol — transcription_delay/end_of_turn_delay en el
    # turno del usuario, llm_node_ttft/tts_node_ttfb/e2e_latency en el de MVLL). Se
    # graba tal cual a JSONL para que analyze_latency.py calcule las estadísticas
    # (media, mediana, p95) sin depender de parsear el log de texto.
    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev) -> None:
        message = ev.item
        metrics = getattr(message, "metrics", None)
        if not metrics:
            return
        row = {"timestamp": time.time(), "role": message.role, **metrics}
        with open(LATENCY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if message.role == "assistant" and "e2e_latency" in metrics:
            print(
                f"[Latencia E2E] {metrics['e2e_latency'] * 1000:.0f}ms "
                f"(llm_ttft={metrics.get('llm_node_ttft', 0) * 1000:.0f}ms, "
                f"tts_ttfb={metrics.get('tts_node_ttfb', 0) * 1000:.0f}ms)"
            )

    # Iniciar la sesión con el agente MVLL
    await session.start(
        room=ctx.room,
        agent=MVLLAgent(),
        room_input_options=RoomInputOptions(),
    )

    # Saludo inicial fijo (texto determinístico vía session.say(), sin pasar por el
    # LLM) — disimula los ~3s de warmup de cancelación de eco acústico que arrancan
    # apenas conecta el participante (ver "aec warmup active, disabling interruptions
    # for 3.00s" en el log): sin esto, esos 3s se sienten como que Mario tarda en
    # reaccionar antes incluso de que el usuario diga nada. session.say() sintetiza y
    # reproduce directo, sin ida y vuelta al LLM, así que arranca casi de inmediato.
    session.say(
        "Soy Mario Vargas Llosa, y aquí me tienes, dispuesto a conversar sobre "
        "literatura, libertad y el oficio de escribir."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="mvll_local",
        ),
    )
