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
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.agents.llm import FallbackAdapter
from livekit.plugins import aws as aws_llm
from livekit.plugins import google, elevenlabs, silero

# Modelo Claude vía AWS Bedrock usado como respaldo del LLM. Sonnet 4.6 es el más
# capaz al que esta cuenta de AWS tiene acceso habilitado (Opus 4.8 y Sonnet 5
# devuelven AccessDeniedException — el acceso al modelo no está aprobado todavía).
BEDROCK_FALLBACK_MODEL = "us.anthropic.claude-sonnet-4-6"


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

        # LLM: Claude (AWS Bedrock) primero, con Gemini vía Vertex AI (no AI Studio —
        # facturado a la cuenta de GCP, sin límite diario de cuota gratuita) como
        # respaldo si Bedrock llegara a fallar.
        llm=FallbackAdapter([
            aws_llm.LLM(
                model=BEDROCK_FALLBACK_MODEL,
                api_key=os.getenv("AWS_ACCESS_KEY_ID"),
                api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            ),
            google.LLM(
                model="gemini-2.5-flash",
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
                temperature=0.7,
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
