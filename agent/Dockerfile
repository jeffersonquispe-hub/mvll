# Agente de voz LiveKit de MVLL — contenedor Linux.
# Existe porque esta máquina Windows tiene Smart App Control en modo enforce, que
# bloquea las DLLs sin firmar de av/PyAV (dependencia de livekit-agents). Dentro de
# un contenedor Linux PyAV carga sin problema. El agente solo hace conexiones
# salientes a LiveKit Cloud, así que no expone ningún puerto.

FROM python:3.12-slim

# ffmpeg: por si alguna ruta de PyAV necesita los binarios del sistema.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mvll_agent.py .

# Descarga los pesos de Silero VAD y demás archivos de los plugins en build,
# para que el arranque en runtime sea inmediato.
RUN python mvll_agent.py download-files

# `start` = modo producción del CLI de livekit-agents (no `dev`).
CMD ["python", "mvll_agent.py", "start"]
