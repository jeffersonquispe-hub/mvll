"""
Agente de voz LiveKit para el Avatar Conversacional de Mario Vargas Llosa.

Pipeline: STT (Google) → LLM (Gemini 2.5 Flash) → TTS (ElevenLabs)

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
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env de forma absoluta y configurar credenciales de Google
env_path = Path("C:/Users/user/Desktop/MVLL/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    # Configurar las variables del sistema para LiveKit y ElevenLabs
    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    # Credenciales de Google: STT y LLM (Vertex AI) usan la misma ADC
    # (GOOGLE_APPLICATION_CREDENTIALS), a diferencia de antes cuando el LLM usaba
    # AI Studio con API key y había que retirar la variable del entorno para que
    # no interfiriera. El STT recibe la ruta explícita igualmente.
    google_stt_credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

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
from livekit.agents.llm import LLM, LLMStream, FallbackAdapter
from livekit.plugins import aws as aws_llm
from livekit.plugins import google, elevenlabs, silero

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

    def prewarm(self) -> None:
        self._wrapped.prewarm()

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

## Reglas para conversación de voz
- Responde de forma CONCISA: 2-4 oraciones máximo (es una conversación hablada, no un ensayo)
- Sé directo y ve al grano
- Adapta la extensión al tipo de pregunta: preguntas simples → respuestas cortas
- Habla siempre en español
"""


class MVLLAgent(Agent):
    """Agente conversacional que encarna a Mario Vargas Llosa."""

    def __init__(self):
        super().__init__(
            instructions=MVLL_SYSTEM_PROMPT,
        )


async def entrypoint(ctx: JobContext):
    """Punto de entrada del agente — se ejecuta cuando un participante entra a la sala."""

    await ctx.connect()

    session = AgentSession(
        # Detección de actividad de voz
        vad=silero.VAD.load(),

        # Speech-to-Text en español
        stt=google.STT(
            languages=["es-ES"],
            credentials_file=google_stt_credentials_file,
        ),

        # LLM: Claude (AWS Bedrock) primero, con Gemini vía Vertex AI como respaldo. Cuenta de AWS
        # nueva (435788423620, usuario "voice") — todavía le falta llenar el formulario de "use
        # case details" de Anthropic en la consola de Bedrock, así que por ahora Bedrock puede
        # fallar en cada turno (ResourceNotFoundException) hasta que se complete ese paso. Una vez
        # confirmado que responde de forma estable, no hace falta tocar nada más acá.
        #
        # Ambos LLM van envueltos en TimeoutGuardLLM: FallbackAdapter.attempt_timeout solo cubre
        # conectar + recibir headers (vía aiohttp ClientTimeout(total=...), que deja de vigilar
        # apenas arranca el stream) — si el stream se traba a mitad de generación, hoy nada lo
        # detecta ni dispara el fallback (causó un cuelgue real de 37s en una llamada). El wrapper
        # le pone un límite a cada fragmento del stream, no solo a la conexión inicial.
        llm=FallbackAdapter([
            TimeoutGuardLLM(
                aws_llm.LLM(
                    model=BEDROCK_VOICE_MODEL,
                    api_key=os.getenv("AWS_ACCESS_KEY_ID"),
                    api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                ),
                chunk_timeout=LLM_CHUNK_STALL_TIMEOUT_S,
            ),
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
        ]),

        # Text-to-Speech: ElevenLabs con la voz clonada de MVLL
        tts=elevenlabs.TTS(
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "7B1CbnTtwwTp1CCGjRzn"),
            model="eleven_flash_v2_5",
            language="es",
        ),
    )

    # Iniciar la sesión con el agente MVLL
    await session.start(
        room=ctx.room,
        agent=MVLLAgent(),
        room_input_options=RoomInputOptions(),
    )

    # Saludo inicial deshabilitado temporalmente para aislar pruebas del loop
    # STT -> LLM -> TTS por turno del usuario. Para reactivarlo, descomentar:
    # await session.generate_reply(
    #     instructions="Preséntate brevemente como Mario Vargas Llosa en una sola oración corta. No digas 'hola'. Empieza directo con algo como 'Soy Mario Vargas Llosa, y estoy aquí para conversar sobre literatura, libertad y el oficio de escribir.'"
    # )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="mvll_local",
        ),
    )
