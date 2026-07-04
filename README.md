# Avatar Conversacional de Mario Vargas Llosa (MVLL) - MVP

Este es el Producto Mínimo Viable (MVP) para la capa de orquestación inteligente y síntesis de voz (TTS) de un avatar interactivo del premio Nobel **Mario Vargas Llosa**.

El sistema procesa texto o voz del visitante, genera con Gemini la respuesta de Mario Vargas Llosa (como escritor e intelectual político liberal) directamente en su característica prosa literaria, y genera un archivo de audio con su voz clonada usando ElevenLabs.

---

## 🛠️ Estructura del Proyecto

```text
MVLL/
│
├── backend/                    # Backend en Python (FastAPI) — único backend HTTP
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # Servidor FastAPI (chat, streaming, ASR, token de LiveKit)
│   │   ├── config.py           # Carga de variables de entorno y directorios
│   │   ├── agents.py           # Generación de la respuesta del avatar (Bedrock/Claude → Gemini)
│   │   ├── bedrock_fallback.py # Respaldo del LLM: Claude vía AWS Bedrock
│   │   ├── tts.py              # Síntesis de voz y caché local (ElevenLabs)
│   │   ├── asr.py              # Reconocimiento de voz (Google Cloud STT → AWS Transcribe)
│   │   └── transcribe_fallback.py  # Respaldo del ASR: AWS Transcribe
│   │
│   ├── public/cache/           # Audios generados, cacheados por hash del texto
│   ├── run.py                  # Script de inicio del backend
│   └── requirements.txt        # Librerías de Python requeridas
│
├── agent/
│   ├── mvll_agent.py           # Agente de voz en tiempo real (LiveKit: STT → LLM → TTS)
│   └── requirements.txt
│
├── frontend/
│   ├── index.html              # Interfaz web de usuario
│   ├── style.css               # Estilos clásicos (Burgundy & Parchment)
│   ├── app.js                  # Lógica del cliente, ASR local y visualizaciones
│   └── mario.png               # Retrato artístico del avatar de MVLL
│
├── samples/                    # Audios de muestra para probar el ASR
│   ├── test_voice.mp3
│   └── test_voice_es.mp3
│
├── start-backend.cmd           # Arranca el backend sin depender de la carpeta/venv activo
├── start-agent.cmd             # Arranca el agente de voz (con auto-reinicio si se cae)
├── .env                        # Único archivo de credenciales del proyecto
├── .gitignore
├── CLAUDE.md                   # Guía del repo para agentes de código
└── README.md                   # Este archivo
```

---

## 🚀 Requisitos de Sistema

1. **Python 3.10+**: se administra con la herramienta `uv` (entorno virtual + dependencias, sin necesitar `pip` global).
2. **Claves de API**:
   * **Gemini API Key**: necesaria para que el avatar genere respuestas.
   * **ElevenLabs API Key**: para la síntesis de voz del timbre real de MVLL (Voice ID: `7B1CbnTtwwTp1CCGjRzn`).
   * **Google Cloud Credentials (opcional)**: solo necesario si querés usar el ASR en el backend en lugar del ASR nativo del navegador.
   * **LiveKit Cloud (opcional)**: solo necesario para la llamada de voz en tiempo real (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`).

---

## ⚙️ Configuración e Instalación

### Backend (FastAPI)

Usa la herramienta de empaquetado de alto rendimiento `uv` para levantar el entorno virtual e instalar las dependencias sin `pip` global:

1. Abre la consola en la carpeta `backend`:
   ```powershell
   cd backend
   ```
2. Crea el entorno virtual de Python administrado por `uv`:
   ```powershell
   uv venv
   ```
3. Activa el entorno virtual en PowerShell:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
4. Instala las dependencias en el entorno virtual usando `uv`:
   ```powershell
   uv pip install -r requirements.txt
   ```
5. Completá tus llaves en el `.env` de la raíz del proyecto (`MVLL/.env`): `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`, etc. Es el único `.env` del proyecto — lo usan tanto el backend como el agente de voz.
6. Inicia el servidor:
   ```powershell
   .\.venv\Scripts\python.exe run.py
   ```
   (Si activaste el entorno virtual en el paso 3 y el prompt muestra `(.venv)`, alcanza con `python run.py`. Si no tenés Python instalado globalmente, `python run.py` a secas puede abrir el instalador de la Microsoft Store en vez de correr el script — en ese caso usá el comando de arriba, que invoca el Python del entorno virtual directamente.)
7. Abre en tu navegador [http://localhost:8000](http://localhost:8000) (o el puerto configurado en `PORT` dentro del `.env`).

**Después de esta instalación inicial**, para arrancar el backend usá directamente `start-backend.cmd` en la raíz del proyecto (doble click, o `.\start-backend.cmd` desde PowerShell) — ya resuelve la carpeta y el Python del entorno virtual por vos, sin importar desde dónde lo corras.

### Agente de voz en tiempo real (opcional, LiveKit)

Para conversar por voz en tiempo real (en vez de mensajes de texto), corré el agente de LiveKit como un proceso aparte:

```powershell
cd agent
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
python mvll_agent.py dev
```

**Después de esta instalación inicial**, para arrancar el agente usá directamente `start-agent.cmd` en la raíz del proyecto, con el mismo criterio que `start-backend.cmd`.

Este agente lee sus credenciales del mismo `.env` de la raíz del proyecto (`MVLL/.env`) que usa el backend.

---

## 🎯 Características Destacadas del MVP

1. **Respuesta del avatar en una sola llamada a Gemini:**
   * El modelo razona internamente sobre la postura, los argumentos conceptuales y las referencias literarias/filosóficas que Vargas Llosa usaría, y redacta directamente la respuesta final en su estilo lírico clásico (oraciones largas con cláusulas complejas coordinadas por comas/puntos y coma, vocabulario intelectual y voz en primera persona) — sin pasos intermedios.
2. **ASR Híbrido (Speech-To-Text):**
   * Por defecto utiliza la API nativa de audio del navegador (`webkitSpeechRecognition`), que es **gratuita, en tiempo real e instantánea**.
   * Opcionalmente, se puede activar la casilla en la interfaz para grabar en formato de audio y transcribir usando el SDK de **Google Cloud Speech-to-Text** en el backend.
3. **Caché Local de Voz (Latencia <10ms):**
   * El backend procesa las respuestas mediante un hash SHA256 único. Si una respuesta ya ha sido hablada por el avatar, el archivo de audio se lee inmediatamente del disco local en lugar de llamar a ElevenLabs. Esto reduce el tiempo de respuesta del habla a **menos de 10ms**, ahorrando créditos de API.
4. **Streaming de texto y audio en paralelo:**
   * El chat de texto usa un endpoint de streaming (`/api/chat/stream-v2`) que transmite la respuesta palabra por palabra mientras Gemini la genera, y envía cada oración completa a ElevenLabs por WebSocket para reproducir el audio por partes, sin esperar a que termine la respuesta completa.
5. **Llamada de voz en tiempo real (LiveKit):**
   * El botón de videollamada del frontend conecta por WebRTC a una sala de LiveKit Cloud atendida por `agent/mvll_agent.py`, un pipeline STT → LLM → TTS que conversa en vivo.
6. **Modo de Simulación (Mock Mode):**
   * Si no tenés conexión a internet o claves a la mano, el sistema cae automáticamente en "Modo Simulación": respuestas de respaldo en español y el sintetizador nativo del navegador (`window.speechSynthesis`) para hablar, permitiendo probar toda la UI del MVP sin credenciales.
7. **Lip-Sync Visual:**
   * La animación de oscilación de ondas del avatar se sincroniza dinámicamente con la reproducción del audio (streaming o WebRTC) o con el sintetizador del navegador.
