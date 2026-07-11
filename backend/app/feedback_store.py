import httpx
from app.config import settings

def is_supabase_configured() -> bool:
    return bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY)

async def save_feedback_supabase(message: str) -> bool:
    """Inserta una fila en la tabla `feedback` de Supabase vía su API REST (PostgREST).
    Usa la service_role key, que bypassa RLS. Devuelve False si falla en vez de lanzar,
    para que el endpoint pueda caer al respaldo local en disco."""
    url = f"{settings.SUPABASE_URL}/rest/v1/feedback"
    headers = {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, headers=headers, json={"message": message})
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error guardando feedback en Supabase: {e}")
        return False
