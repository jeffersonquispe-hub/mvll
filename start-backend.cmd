@echo off
REM Levanta el backend (FastAPI) sin importar desde donde lo corras ni si
REM tenes Python instalado globalmente: usa siempre el venv de backend/.
cd /d "%~dp0backend"
".venv\Scripts\python.exe" run.py
pause
