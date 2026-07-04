import os
import asyncio
from google.cloud import speech
from app.config import settings
from app.transcribe_fallback import is_transcribe_configured, transcribe_via_aws

def is_google_asr_configured() -> bool:
    """Verifica si las credenciales de Google Cloud ASR están configuradas."""
    creds_path = settings.GOOGLE_APPLICATION_CREDENTIALS or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    return bool(creds_path and os.path.exists(creds_path))

def _transcribe_google_sync(audio_content: bytes, content_type: str) -> str:
    client = speech.SpeechClient()

    # Determinar el tipo de codificación según el content_type
    # El navegador suele enviar audio/webm o audio/ogg; audio/mpeg y audio/mp3
    # son ambos MIME types válidos para MP3.
    content_type = content_type.lower()
    if "wav" in content_type:
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        sample_rate_hertz = 16000
    elif "mp3" in content_type or "mpeg" in content_type:
        encoding = speech.RecognitionConfig.AudioEncoding.MP3
        sample_rate_hertz = 44100
    else:
        encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        sample_rate_hertz = 48000

    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate_hertz,
        language_code="es-ES",
        enable_automatic_punctuation=True
    )

    # Para audios de corta duración del MVP, usamos recognize (síncrono)
    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript.strip()

async def transcribe_audio(audio_content: bytes, content_type: str = "audio/webm") -> str:
    """
    Transcribe el audio utilizando Google Cloud Speech-to-Text. Si no está configurado
    o falla, cae a AWS Transcribe como respaldo. Ambos SDKs son síncronos, así que
    corren en un hilo aparte para no bloquear el event loop de FastAPI.
    """
    if is_google_asr_configured():
        try:
            transcript = await asyncio.to_thread(_transcribe_google_sync, audio_content, content_type)
            if transcript:
                return transcript
        except Exception as e:
            print(f"Error en Google ASR: {e}")
    else:
        print("Google Cloud Speech-to-Text no configurado o sin archivo de credenciales.")

    if is_transcribe_configured():
        try:
            transcript = await asyncio.to_thread(transcribe_via_aws, audio_content, content_type)
            if transcript:
                return transcript
        except Exception as e:
            print(f"Error en AWS Transcribe (fallback): {e}")

    return ""
