import os
from dotenv import load_dotenv

# Cargar el .env de la raíz del proyecto (el mismo que usa agent/mvll_agent.py),
# para no mantener credenciales duplicadas en dos archivos separados.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(base_dir)
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

class Settings:
    PORT: int = int(os.getenv("PORT", 8000))
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "7B1CbnTtwwTp1CCGjRzn")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_REGION: str = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "")
    LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    AWS_S3_ASR_BUCKET: str = os.getenv("AWS_S3_ASR_BUCKET", "")

    # Directorio para guardar archivos de audio temporales o cacheados
    CACHE_DIR: str = os.path.join(base_dir, "public", "cache")

    # Archivo donde se acumula el feedback enviado desde la ventana "Sobre el proyecto"
    # (fuera de public/, no se sirve estáticamente)
    FEEDBACK_FILE: str = os.path.join(base_dir, "data", "feedback.jsonl")

settings = Settings()

# Asegurar que el directorio de caché existe
os.makedirs(settings.CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.FEEDBACK_FILE), exist_ok=True)
