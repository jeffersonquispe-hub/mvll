import boto3
from app.config import settings

# Sonnet 4.6 vía perfil de inferencia multi-región: es el modelo Claude más
# capaz al que esta cuenta de AWS tiene acceso habilitado en Bedrock (Opus 4.8
# y Sonnet 5 devuelven AccessDeniedException — el acceso al modelo no está
# aprobado para esta cuenta todavía).
MODEL_ID = "us.anthropic.claude-sonnet-4-6"

def is_bedrock_configured() -> bool:
    return bool(settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY)

def _client():
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.AWS_DEFAULT_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

def run_bedrock_response(prompt_text: str) -> str:
    """Genera la respuesta completa con Claude (Bedrock), sin streaming."""
    client = _client()
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt_text}]}],
        inferenceConfig={"maxTokens": 1024},
    )
    return response["output"]["message"]["content"][0]["text"].strip()

def run_bedrock_stream(prompt_text: str):
    """Generador síncrono que produce la respuesta de Claude (Bedrock) en fragmentos de texto."""
    client = _client()
    response = client.converse_stream(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt_text}]}],
        inferenceConfig={"maxTokens": 1024},
    )
    for event in response["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            yield delta["text"]
