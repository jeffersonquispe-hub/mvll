from google import genai
from app.config import settings
from app.bedrock_fallback import is_bedrock_configured, run_bedrock_response, run_bedrock_stream

# Respuesta de respaldo cuando no hay credenciales de Gemini o la llamada falla
FALLBACK_RESPONSE = (
    "Siempre he creído que la literatura es un fuego que se alimenta de la insatisfacción; "
    "un artificio soberbio donde los seres humanos buscamos aquella verdad secreta que la realidad cotidiana "
    "insiste en negarnos. Quienes pretendan silenciar la voz creadora o imponer dogmas a la inteligencia "
    "olvidan que una sociedad libre solo respira a través del disenso, de la duda metódica y de esa disciplinada "
    "herejía que llamamos ficción."
)

_vertex_client = None

def is_vertex_configured() -> bool:
    return bool(settings.GOOGLE_CLOUD_PROJECT and settings.GOOGLE_APPLICATION_CREDENTIALS)

def get_vertex_client():
    """Cliente de Gemini vía Vertex AI (facturado a la cuenta de GCP, no a los
    créditos prepago de AI Studio). Usa ADC (GOOGLE_APPLICATION_CREDENTIALS) para
    autenticarse, igual que el Speech-to-Text."""
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_REGION,
        )
    return _vertex_client

def get_model_name():
    return "gemini-2.5-flash"

def build_fused_prompt(prompt: str) -> str:
    """Prompt fusionado: el modelo razona internamente sobre postura, argumentos y
    referencias (lo que antes hacía un 'Agente Director' en una llamada separada) pero
    responde en una sola pasada, ya en la prosa literaria final de Mario Vargas Llosa."""
    return f"""
Eres Mario Vargas Llosa, intelectual peruano de convicciones liberales clásicas profundas: crees en la libertad individual, la democracia y la separación de poderes, y criticas severamente los autoritarismos (de izquierda o derecha). Como escritor, estás obsesionado con el poder de la ficción como rebelión contra la realidad, la disciplina del oficio, y tus obras (como 'Conversación en La Catedral', 'La fiesta del Chivo', 'La ciudad y los perros', 'La verdad de las mentiras').

Antes de responder, piensa en silencio (sin escribirlo) cuál es tu postura ante la pregunta, 2 o 3 argumentos conceptuales que la sustenten, y referencias concretas a tus novelas, ensayos o a otros pensadores/escritores afines (Flaubert, Sartre, Popper). Luego redacta directamente la respuesta final incorporando esas ideas, sin enumerarlas ni exponer el razonamiento.

Sigue estas directrices estilísticas estrictas al redactar:
1. **Sintaxis de Cláusulas Largas y Fluidas:** Usa frases amplias y cadenciosas, coordinadas y subordinadas con comas y puntos y coma. Evita las oraciones cortas e inconexas de los asistentes virtuales estándar.
2. **Vocabulario Intelectual y Evocativo:** Utiliza términos literarios y filosóficos propios de tu léxico (ej. 'ficción', 'artificio', 'desvarío', 'fanatismo', 'verdad de las mentiras', 'sartreano', 'cataclismo', 'dictador', 'escribidor', 'disciplina implacable').
3. **Voz en Primera Persona:** Habla siempre en primera persona del singular ('Yo', 'A lo largo de mi vida...', 'Siempre he sostenido...').
4. **Sin Fórmulas Artificiales de Chatbot:** NUNCA uses saludos genéricos como '¡Hola! ¿En qué te puedo ayudar hoy?' o cierres mecánicos de asistente. Empieza directamente con la reflexión literaria o política.
5. **Profundidad y Pasión:** Transmite la pasión por el oficio de escribir como un acto de rebelión y la defensa vehemente de la libertad política frente a la tiranía.
6. **Extensión:** La respuesta debe tener entre 3 y 5 oraciones largas, formando un texto compacto, lírico e intelectual de unas 100-150 palabras, ideal para la síntesis de voz posterior.

Pregunta del usuario: {prompt}

Redacta tu respuesta literaria final:
    """

def run_response(prompt: str) -> str:
    """
    Genera la respuesta final del avatar de Mario Vargas Llosa en una sola llamada al LLM.
    Intenta Claude vía AWS Bedrock primero; si no está configurado o falla, cae a Gemini
    vía Vertex AI (facturado a la cuenta de GCP, no a AI Studio); si eso también falla,
    usa un texto fijo.
    """
    if is_bedrock_configured():
        try:
            return run_bedrock_response(build_fused_prompt(prompt))
        except Exception as e:
            print(f"Error generando la respuesta del avatar (Bedrock): {e}")

    if is_vertex_configured():
        try:
            response = get_vertex_client().models.generate_content(
                model=get_model_name(),
                contents=build_fused_prompt(prompt),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generando la respuesta del avatar (fallback Vertex Gemini): {e}")

    return FALLBACK_RESPONSE

def run_response_stream(prompt: str):
    """
    Generador síncrono que produce la respuesta del avatar en fragmentos de texto
    (usado por los endpoints SSE). Pensado para correr dentro de un hilo (ver main.py),
    ya que tanto boto3 como el SDK de google-genai son síncronos.

    Intenta Claude vía AWS Bedrock primero. Si no llega a producir ningún fragmento (no
    configurado o falla antes del primer chunk), cae por completo a Gemini vía Vertex AI
    en vez de mezclar texto de dos proveedores en la misma respuesta.
    """
    if is_bedrock_configured():
        yielded_any = False
        try:
            for piece in run_bedrock_stream(build_fused_prompt(prompt)):
                yielded_any = True
                yield piece
        except Exception as e:
            print(f"Error generando la respuesta del avatar (stream, Bedrock): {e}")
        if yielded_any:
            return

    if is_vertex_configured():
        yielded_any = False
        try:
            stream = get_vertex_client().models.generate_content_stream(
                model=get_model_name(),
                contents=build_fused_prompt(prompt),
            )
            for chunk in stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
        except Exception as e:
            print(f"Error generando la respuesta del avatar (stream, fallback Vertex Gemini): {e}")
        if yielded_any:
            return

    for word in FALLBACK_RESPONSE.split(" "):
        yield word + " "
