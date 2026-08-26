import re
from google import genai
from app.config import settings
from app.bedrock_fallback import is_bedrock_configured, run_bedrock_response, run_bedrock_stream

# Gemini 2.5 Flash Lite a veces escribe el razonamiento que el prompt le pide pensar
# "en silencio" como texto plano (ej. "(Silencio mental)\nPostura: ...", "(Piensa:
# ...)") en vez de omitirlo del todo. El prompt ya lo prohíbe explícitamente (ver
# build_fused_prompt), pero eso solo reduce la frecuencia — este filtro es la
# garantía a nivel de código, mismo mecanismo que agent/mvll_agent.py
# (MVLLAgent.llm_node): en todos los casos vistos hasta ahora la respuesta
# problemática arranca con "(" y el contenido real viene después de la primera
# línea en blanco, así que se filtra por ese patrón en vez de perseguir etiquetas
# puntuales (el modelo inventa una nueva cada vez).
#
# "Respuesta" es un caso aparte: a diferencia de "Postura"/"Argumento"/"Referencia"
# (que son razonamiento puro, se descartan enteras) el modelo a veces la usa como
# prefijo de la respuesta REAL en la misma línea (ej. "Respuesta: Afortunadamente,
# estos días..." con un solo salto de línea antes, sin línea en blanco) — si se
# tratara igual que las demás, se perdería la respuesta entera, no solo la etiqueta.
_DROP_LABEL = r"(?:Silencio mental|Postura|Argumento|Referencia)(?:\s*\d+)?"
_ANSWER_LABEL = r"Respuesta(?:\s*\d+)?"
_MARKUP = r"[\s\*\-_]*"  # tolera bullets/negrita markdown alrededor de la etiqueta

_REASONING_LEAK_RE = re.compile(
    rf"^{_MARKUP}\(?{_MARKUP}{_DROP_LABEL}{_MARKUP}\)?{_MARKUP}:"  # "Postura: ..." -> línea entera fuera
    rf"|^{_MARKUP}\({_MARKUP}{_DROP_LABEL}{_MARKUP}\){_MARKUP}$"  # "(Silencio mental)" sola -> fuera
    rf"|^{_MARKUP}\({_MARKUP}{_ANSWER_LABEL}{_MARKUP}\){_MARKUP}$",  # "(Respuesta)" sola (sin contenido) -> fuera
    re.IGNORECASE,
)
_ANSWER_PREFIX_RE = re.compile(
    rf"^{_MARKUP}\(?{_MARKUP}{_ANSWER_LABEL}{_MARKUP}\)?{_MARKUP}:{_MARKUP}",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str | None:
    """None si la línea es puro razonamiento a descartar; si no, la línea (con el
    prefijo "Respuesta:" recortado si lo tenía, conservando el contenido real que
    sigue en la misma línea)."""
    if _REASONING_LEAK_RE.match(line):
        return None
    return _ANSWER_PREFIX_RE.sub("", line, count=1)


def _clean_lines(text: str) -> str:
    cleaned = (_clean_line(line) for line in text.split("\n"))
    return "\n".join(line for line in cleaned if line is not None)


# Ventana (en caracteres) donde se busca la primera línea en blanco al arrancar
# una respuesta. Antes esto solo se activaba si el texto empezaba con "(", pero el
# modelo también se filtró sin paréntesis (ej. "Charla cotidiana.\n\n¡Ah, la
# carapulcra!...", repitiendo literal el nombre de la categoría del PASO 1 del
# prompt) — así que ahora se vigila SIEMPRE, sin condicionarlo al primer carácter.
# El prompt nunca pide una respuesta con línea en blanco interna (es prosa continua
# de un párrafo), así que cualquier "\n\n" temprano es señal de preámbulo, venga
# como venga. Acotado a una ventana corta para no demorar el primer token de una
# respuesta limpia (que jamás va a tener "\n\n", así que siempre paga la ventana
# completa antes de empezar a fluir).
_PREAMBLE_WATCH_WINDOW = 120

# Este prompt (a diferencia del de agent/mvll_agent.py) le pide al modelo clasificar
# el mensaje en silencio como "(A)" o "(B)" (ver PASO 1 en build_fused_prompt) — a
# veces repite esa etiqueta pegada directo al inicio de la respuesta real, sin línea
# en blanco que las separe (ej. "(B) El populismo en América Latina es..."). Se
# filtra aparte porque no encaja en el patrón de preámbulo-hasta-línea-en-blanco.
_CLASSIFICATION_TAG_RE = re.compile(r"^\(\s*[A-Za-z]\)\s*")


def _strip_reasoning_preamble(text: str) -> str:
    """Filtra, de una respuesta ya completa, la etiqueta de clasificación, el
    preámbulo de razonamiento (si aparece una línea en blanco temprano) y cualquier
    línea suelta con una etiqueta conocida."""
    text = _CLASSIFICATION_TAG_RE.sub("", text.lstrip(), count=1)
    idx = text.find("\n\n")
    if idx != -1 and idx < _PREAMBLE_WATCH_WINDOW:
        text = text[idx + 2 :]
    # Filtrado línea por línea siempre, no solo cuando hubo línea en blanco: si el
    # preámbulo nunca tiene "\n\n" (ej. "(Silencio mental)\nPostura: ...\nRespuesta:
    # ...", todo con saltos simples), el texto entero llega hasta acá sin recortar,
    # y son las etiquetas sueltas las que terminan de limpiarlo.
    return _clean_lines(text).strip()


def _filter_reasoning_leak_stream(chunks):
    """Envoltorio de streaming: aplica el mismo filtro que _strip_reasoning_preamble
    incrementalmente, a medida que llegan los fragmentos, en vez de esperar a que
    termine toda la respuesta."""
    buffer = ""
    tag_checked = False  # todavía no se decidió si hay etiqueta de clasificación
    watching_preamble = True  # buscando "\n\n" temprano, ver _PREAMBLE_WATCH_WINDOW
    for piece in chunks:
        buffer += piece

        if not tag_checked:
            # Esperar unos caracteres más antes de decidir: "(B) " puede llegar
            # repartido en varios fragmentos (ej. "(" solo en el primer chunk).
            if len(buffer) < 5 and "\n" not in buffer:
                continue
            tag_checked = True
            buffer = _CLASSIFICATION_TAG_RE.sub("", buffer, count=1)

        if watching_preamble:
            idx = buffer.find("\n\n")
            if idx != -1:
                buffer = buffer[idx + 2 :]
                watching_preamble = False
            elif len(buffer) < _PREAMBLE_WATCH_WINDOW:
                continue
            else:
                watching_preamble = False

        if "\n" not in buffer:
            continue
        *complete_lines, buffer = buffer.split("\n")
        cleaned = "\n".join(
            line for line in (_clean_line(l) for l in complete_lines) if line is not None
        )
        if cleaned:
            yield cleaned + "\n"

    if not tag_checked:
        buffer = _CLASSIFICATION_TAG_RE.sub("", buffer, count=1)
    if watching_preamble:
        idx = buffer.find("\n\n")
        if idx != -1:
            buffer = buffer[idx + 2 :]
    # Filtrado línea por línea siempre (no un simple match() sobre todo el buffer):
    # si la respuesta entera nunca tuvo "\n\n" ni superó la ventana, watching_preamble
    # queda True hasta acá y el buffer trae todo el texto crudo con saltos simples
    # entre etiquetas — hay que partirlo por línea para que cada etiqueta se filtre.
    cleaned = _clean_lines(buffer)
    if cleaned:
        yield cleaned


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
    return "gemini-2.5-flash-lite"

def build_fused_prompt(prompt: str) -> str:
    """Prompt fusionado: el modelo razona internamente sobre postura, argumentos y
    referencias (lo que antes hacía un 'Agente Director' en una llamada separada) pero
    responde en una sola pasada, ya en la prosa literaria final de Mario Vargas Llosa."""
    return f"""
Vas a responder, como Mario Vargas Llosa, al siguiente mensaje de un usuario: "{prompt}"

PASO 1 — Clasifica ese mensaje, en silencio, en una sola categoría:
(A) Charla cotidiana: saludo, "¿cómo estás?", agradecimiento, despedida, comentario casual — CUALQUIER mensaje sin una pregunta o afirmación sustantiva sobre literatura, política, filosofía o tu obra cae aquí, aunque te esté hablando a ti (Mario Vargas Llosa) directamente.
(B) Solo si hay una pregunta o planteo real con sustancia intelectual: literatura, política, filosofía, tu obra, actualidad, sociedad.

Ante la duda entre A y B, elige A.

PASO 2 — Si clasificaste (A):
Tu respuesta ENTERA debe tener como máximo 40 palabras (puede ser más corta si con eso alcanza — no la alargues de relleno). Tono cálido y humano, como hablaría cualquier persona, pero con algo de tu color personal: podés soltar una observación ingeniosa, una pizca de ironía o una imagen breve que se sienta tuya, sin caer en el discurso ni en párrafos. PROHIBIDO en este caso: vocabulario grandilocuente ('cataclismo', 'artificio', 'fanatismo', 'desvarío', etc.), cláusulas largas con punto y coma, y referencias extensas a tu obra o a otros escritores. Si te saluda, salúdalo de vuelta con naturalidad — lo que hay que evitar es el saludo hueco de chatbot ('¡Hola! ¿En qué puedo ayudarte hoy?'), no el saludo humano.
Ejemplo de la longitud y el tono correctos — Usuario: "Hola Mario, ¿cómo estás?" → Respuesta completa: "Muy bien, aquí, entre libros como siempre. ¿Y tú, qué cuentas?" (así de breve está bien; usa las 40 palabras solo si de verdad suman algo, no por llenar espacio).
Antes de dar tu respuesta final, revisa: si pasa las 40 palabras o usa vocabulario literario grandilocuente, recortala.

PASO 3 — Si clasificaste (B), asume tu identidad completa:
Eres Mario Vargas Llosa, intelectual peruano de convicciones liberales clásicas profundas: crees en la libertad individual, la democracia y la separación de poderes, y criticas severamente los autoritarismos (de izquierda o derecha). Como escritor, estás obsesionado con el poder de la ficción como rebelión contra la realidad, la disciplina del oficio, y tus obras (como 'Conversación en La Catedral', 'La fiesta del Chivo', 'La ciudad y los perros', 'La verdad de las mentiras'). Piensa cuál es tu postura, 2 o 3 argumentos conceptuales que la sustenten, y referencias concretas a tus novelas, ensayos o a otros pensadores/escritores afines (Flaubert, Sartre, Popper); luego redacta la respuesta final incorporando esas ideas, sin enumerarlas ni exponer el razonamiento. Sigue estas directrices estilísticas estrictas:
    1. **Sintaxis de Cláusulas Largas y Fluidas:** Usa frases amplias y cadenciosas, coordinadas y subordinadas con comas y puntos y coma. Evita las oraciones cortas e inconexas de los asistentes virtuales estándar.
    2. **Vocabulario Intelectual y Evocativo:** Utiliza términos literarios y filosóficos propios de tu léxico (ej. 'ficción', 'artificio', 'desvarío', 'fanatismo', 'verdad de las mentiras', 'sartreano', 'cataclismo', 'dictador', 'escribidor', 'disciplina implacable').
    3. **Profundidad y Pasión:** Transmite la pasión por el oficio de escribir como un acto de rebelión y la defensa vehemente de la libertad política frente a la tiranía.
    4. **Extensión:** La respuesta debe tener entre 3 y 5 oraciones largas, formando un texto compacto, lírico e intelectual de unas 100-150 palabras, ideal para la síntesis de voz posterior.

En ambos casos: habla siempre en primera persona del singular ('Yo', 'A lo largo de mi vida...', 'Siempre he sostenido...'), y nunca uses cierres mecánicos de asistente virtual ('¿Hay algo más en lo que pueda ayudarte?').

REGLA ABSOLUTA: tu respuesta debe ser ÚNICAMENTE el texto final, empezando directo con él. Nunca escribas
etiquetas, encabezados ni marcadores de tu proceso interno — cosas como "(Silencio mental)", "Postura:",
"Argumento:", "Referencia:" o "(Respuesta)" — ni siquiera para separar el razonamiento de la respuesta.

Redacta ahora tu respuesta final (solo la respuesta, sin mencionar el paso ni la categoría):
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
            return _strip_reasoning_preamble(run_bedrock_response(build_fused_prompt(prompt)))
        except Exception as e:
            print(f"Error generando la respuesta del avatar (Bedrock): {e}")

    if is_vertex_configured():
        try:
            response = get_vertex_client().models.generate_content(
                model=get_model_name(),
                contents=build_fused_prompt(prompt),
            )
            return _strip_reasoning_preamble(response.text.strip())
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
            for piece in _filter_reasoning_leak_stream(run_bedrock_stream(build_fused_prompt(prompt))):
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
            raw_pieces = (chunk.text for chunk in stream if chunk.text)
            for piece in _filter_reasoning_leak_stream(raw_pieces):
                yielded_any = True
                yield piece
        except Exception as e:
            print(f"Error generando la respuesta del avatar (stream, fallback Vertex Gemini): {e}")
        if yielded_any:
            return

    for word in FALLBACK_RESPONSE.split(" "):
        yield word + " "
