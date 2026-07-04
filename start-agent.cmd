@echo off
REM Levanta el agente de voz de LiveKit sin importar desde donde lo corras ni si
REM tenes Python instalado globalmente: usa siempre el venv de agent/. Se reinicia
REM solo si el proceso se cae (Ctrl+C para salir de verdad).
cd /d "%~dp0agent"
:loop
echo [%date% %time%] Iniciando agente LiveKit de MVLL...
".venv\Scripts\python.exe" mvll_agent.py dev
echo [%date% %time%] El agente se detuvo. Reiniciando en 2 segundos... (Ctrl+C para salir)
timeout /t 2 >nul
goto loop
