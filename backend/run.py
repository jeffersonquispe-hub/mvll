import uvicorn
import os
import sys

if __name__ == "__main__":
    # Asegurar que el path incluya el directorio actual para encontrar 'app'
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    port = int(os.getenv("PORT", 8000))
    print(f"Iniciando el servidor del avatar de MVLL en http://localhost:{port}...")
    # reload_dirs acotado a app/: sin esto, uvicorn vigila todo backend/ (incluido
    # .venv, con decenas de miles de archivos) y dispara reinicios constantes.
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True, reload_dirs=["app"])
