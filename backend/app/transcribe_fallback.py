import json
import time
import uuid
import boto3
from app.config import settings

def is_transcribe_configured() -> bool:
    return bool(
        settings.AWS_ACCESS_KEY_ID
        and settings.AWS_SECRET_ACCESS_KEY
        and settings.AWS_S3_ASR_BUCKET
    )

def _clients():
    kwargs = dict(
        region_name=settings.AWS_DEFAULT_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    return boto3.client("s3", **kwargs), boto3.client("transcribe", **kwargs)

def _media_format(content_type: str) -> str:
    content_type = (content_type or "").lower()
    if "wav" in content_type:
        return "wav"
    if "mp3" in content_type or "mpeg" in content_type:
        return "mp3"
    if "ogg" in content_type:
        return "ogg"
    return "webm"

def transcribe_via_aws(audio_bytes: bytes, content_type: str = "audio/webm", timeout_seconds: int = 30) -> str:
    """
    Transcribe el audio usando AWS Transcribe (fallback de Google Cloud STT).
    AWS Transcribe no tiene una API síncrona: sube el audio a S3, arranca un job
    de transcripción, espera a que termine (con sondeo) y lee el resultado.
    """
    s3, transcribe = _clients()
    bucket = settings.AWS_S3_ASR_BUCKET
    media_format = _media_format(content_type)
    job_id = uuid.uuid4().hex
    audio_key = f"uploads/{job_id}.{media_format}"
    result_key = f"results/{job_id}.json"
    job_name = f"mvll-asr-{job_id}"

    s3.put_object(Bucket=bucket, Key=audio_key, Body=audio_bytes)

    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode="es-ES",
            MediaFormat=media_format,
            Media={"MediaFileUri": f"s3://{bucket}/{audio_key}"},
            OutputBucketName=bucket,
            OutputKey=result_key,
        )

        deadline = time.time() + timeout_seconds
        status = None
        while time.time() < deadline:
            resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status in ("COMPLETED", "FAILED"):
                break
            time.sleep(1.5)

        if status != "COMPLETED":
            print(f"AWS Transcribe: job no completó a tiempo (status={status})")
            return ""

        result_obj = s3.get_object(Bucket=bucket, Key=result_key)
        result_json = json.loads(result_obj["Body"].read())
        transcripts = result_json.get("results", {}).get("transcripts", [])
        return transcripts[0]["transcript"].strip() if transcripts else ""
    finally:
        # Limpieza inmediata: no hace falta esperar la expiración por lifecycle del bucket.
        for key in (audio_key, result_key):
            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except Exception:
                pass
        try:
            transcribe.delete_transcription_job(TranscriptionJobName=job_name)
        except Exception:
            pass
