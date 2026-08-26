"""
Analiza agent/latency_log.jsonl (generado por mvll_agent.py en cada turno real) y
reporta estadísticas agregadas de latencia end-to-end.

Uso:
    .venv\\Scripts\\python.exe analyze_latency.py [ruta_al_jsonl]
"""

import json
import statistics
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "latency_log.jsonl"

FIELDS = [
    "e2e_latency",
    "transcription_delay",
    "end_of_turn_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "playback_latency",
]

LABELS = {
    "e2e_latency": "Latencia end-to-end (fin de habla del usuario -> inicio de respuesta)",
    "transcription_delay": "STT: demora de transcripción tras fin de habla",
    "end_of_turn_delay": "EOT: demora en decidir que el usuario terminó de hablar",
    "llm_node_ttft": "LLM: time-to-first-token",
    "tts_node_ttfb": "TTS: time-to-first-byte",
    "playback_latency": "Reproducción: latencia de playback",
}


def _stats(values_ms: list[float]) -> dict | None:
    if not values_ms:
        return None
    values_ms = sorted(values_ms)
    n = len(values_ms)
    p95_idx = min(n - 1, round(0.95 * (n - 1)))
    return {
        "n": n,
        "mean": statistics.mean(values_ms),
        "median": statistics.median(values_ms),
        "p95": values_ms[p95_idx],
        "min": values_ms[0],
        "max": values_ms[-1],
    }


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_PATH
    if not path.exists():
        print(f"No existe {path}. Hacé al menos una llamada de prueba real primero.")
        return

    series: dict[str, list[float]] = {f: [] for f in FIELDS}
    n_rows = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            for field in FIELDS:
                value = row.get(field)
                if isinstance(value, (int, float)):
                    series[field].append(value * 1000)

    print(f"Analizando {path} ({n_rows} turnos registrados)\n")

    for field in FIELDS:
        s = _stats(series[field])
        label = LABELS[field]
        if s is None:
            print(f"- {label}: sin datos")
            continue
        print(
            f"- {label}\n"
            f"    n={s['n']}  media={s['mean']:.0f}ms  mediana={s['median']:.0f}ms  "
            f"p95={s['p95']:.0f}ms  min={s['min']:.0f}ms  max={s['max']:.0f}ms"
        )


if __name__ == "__main__":
    main()
