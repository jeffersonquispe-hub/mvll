import os
import hashlib
import httpx
from app.config import settings

async def generate_tts(text: str, api_key: str = None, voice_id: str = None) -> tuple[str, bool]:
    """
    Convierte texto en habla clonada de MVLL usando ElevenLabs.
    Retorna una tupla (ruta_archivo_audio, es_cache_hit).
    """
    # Crear hash único para el texto para usar de nombre de archivo
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"audio_{text_hash}.mp3"
    filepath = os.path.join(settings.CACHE_DIR, filename)

    # 1. Comprobar si ya existe en la caché local para latencia ultra-baja
    if os.path.exists(filepath):
        print("TTS Cache HIT!")
        return filename, True

    print("TTS Cache MISS. Generando audio nuevo...")
    
    # Obtener credenciales
    key = api_key or settings.ELEVENLABS_API_KEY
    v_id = voice_id or settings.ELEVENLABS_VOICE_ID

    # Si no hay credenciales de ElevenLabs, retornamos indicando simulación
    if not key or not v_id:
        print("Sin credenciales de ElevenLabs. Se utilizará simulación en el frontend.")
        return "", False

    # URL del endpoint de ElevenLabs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{v_id}"
    
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "accept": "audio/mpeg"
    }
    
    data = {
        "text": text,
        # eleven_flash_v2_5 es el modelo de baja latencia y optimizado
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.8
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code != 200:
                print(f"Error ElevenLabs ({response.status_code}): {response.text}")
                return "", False
                
            # Guardar el contenido binario del audio
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            return filename, False
            
    except Exception as e:
        print(f"Excepción al llamar a ElevenLabs: {e}")
        return "", False
