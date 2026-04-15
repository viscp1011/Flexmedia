"""
gera_dados.py — Gera dataset simulado realista para o Totem Flexmedia.

Simula 300 sessões de uso ao longo de 30 dias com padrões coerentes:
- 4 perfis de visitante (curioso, engajado, comparador, passante)
- Horários de pico (manhã/tarde/noite) e horários fracos
- Comportamentos distintos por tela e por perfil
- Variação de engajamento entre dias da semana
- Coluna de presença simulada (detecção de pessoa via câmera)

Gera: dados_simulados.csv (~3000 eventos)
"""
import random
import csv
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

SCREENS = ["INICIO", "CATALOGO", "PROMOCOES"]

# ─────────────── Perfis de visitante ───────────────
# Cada perfil tem: matriz de transição, dwell params, n_events, presença
PROFILES = {
    "passante": {
        # Entra, dá uma olhada rápida e sai
        "weight": 0.25,
        "transition": {
            "INICIO":    {"INICIO": 0.10, "CATALOGO": 0.50, "PROMOCOES": 0.40},
            "CATALOGO":  {"INICIO": 0.50, "CATALOGO": 0.10, "PROMOCOES": 0.40},
            "PROMOCOES": {"INICIO": 0.60, "CATALOGO": 0.30, "PROMOCOES": 0.10},
        },
        "dwell": {"INICIO": (1500, 600), "CATALOGO": (3000, 1000), "PROMOCOES": (2500, 800)},
        "n_events": (3, 6),
        "presence": 0.85,
    },
    "curioso": {
        # Explora todas as telas, tempo médio
        "weight": 0.30,
        "transition": {
            "INICIO":    {"INICIO": 0.05, "CATALOGO": 0.60, "PROMOCOES": 0.35},
            "CATALOGO":  {"INICIO": 0.20, "CATALOGO": 0.10, "PROMOCOES": 0.70},
            "PROMOCOES": {"INICIO": 0.25, "CATALOGO": 0.65, "PROMOCOES": 0.10},
        },
        "dwell": {"INICIO": (3000, 1000), "CATALOGO": (7000, 2500), "PROMOCOES": (5000, 1500)},
        "n_events": (5, 10),
        "presence": 0.92,
    },
    "engajado": {
        # Fica muito tempo no CATALOGO e PROMOCOES, alta interação
        "weight": 0.30,
        "transition": {
            "INICIO":    {"INICIO": 0.05, "CATALOGO": 0.70, "PROMOCOES": 0.25},
            "CATALOGO":  {"INICIO": 0.10, "CATALOGO": 0.15, "PROMOCOES": 0.75},
            "PROMOCOES": {"INICIO": 0.15, "CATALOGO": 0.75, "PROMOCOES": 0.10},
        },
        "dwell": {"INICIO": (4000, 1500), "CATALOGO": (12000, 4000), "PROMOCOES": (9000, 3000)},
        "n_events": (8, 15),
        "presence": 0.97,
    },
    "comparador": {
        # Fica alternando entre CATALOGO e PROMOCOES comparando produtos
        "weight": 0.15,
        "transition": {
            "INICIO":    {"INICIO": 0.05, "CATALOGO": 0.75, "PROMOCOES": 0.20},
            "CATALOGO":  {"INICIO": 0.05, "CATALOGO": 0.05, "PROMOCOES": 0.90},
            "PROMOCOES": {"INICIO": 0.05, "CATALOGO": 0.90, "PROMOCOES": 0.05},
        },
        "dwell": {"INICIO": (2000, 800), "CATALOGO": (10000, 3500), "PROMOCOES": (8000, 3000)},
        "n_events": (10, 20),
        "presence": 0.95,
    },
}

# Horários de pico por período
MORNING_PEAK  = list(range(9, 12))   # 9h-11h
AFTERNOON_PEAK = list(range(14, 18)) # 14h-17h
EVENING_PEAK  = list(range(19, 21))  # 19h-20h
ALL_HOURS     = list(range(8, 22))

# Peso de movimento por dia da semana (seg=0 ... dom=6)
DAY_WEIGHTS = [0.8, 0.9, 1.0, 1.0, 1.1, 1.4, 1.2]  # fim de semana mais movimentado


def pick_profile() -> tuple[str, dict]:
    names  = list(PROFILES.keys())
    weights = [PROFILES[n]["weight"] for n in names]
    name   = random.choices(names, weights=weights)[0]
    return name, PROFILES[name]


def next_screen(current: str, profile: dict) -> str:
    probs   = profile["transition"][current]
    screens = list(probs.keys())
    weights = list(probs.values())
    return random.choices(screens, weights=weights)[0]


def generate_dwell(screen: str, profile: dict) -> int:
    mean, std = profile["dwell"][screen]
    return max(300, int(random.gauss(mean, std)))


def pick_session_time(base_date: datetime, day_offset: int) -> datetime:
    """Escolhe horário realista: 75% nos horários de pico."""
    date = base_date + timedelta(days=day_offset)
    day_of_week = date.weekday()
    # Fim de semana tem mais sessões de tarde/noite
    if day_of_week >= 5:
        peak = AFTERNOON_PEAK + EVENING_PEAK
    else:
        peak = MORNING_PEAK + AFTERNOON_PEAK + EVENING_PEAK

    hour   = random.choice(peak) if random.random() < 0.75 else random.choice(ALL_HOURS)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return date.replace(hour=hour, minute=minute, second=second)


def generate_sessions(n_sessions: int = 300) -> list[dict]:
    rows = []
    event_id  = 1
    base_date = datetime(2025, 11, 1, 0, 0, 0)  # 30 dias de novembro de 2025

    for session_idx in range(n_sessions):
        profile_name, profile = pick_profile()

        # Distribui ao longo de 30 dias com peso por dia da semana
        day_offset = random.choices(
            range(30),
            weights=[DAY_WEIGHTS[( base_date + timedelta(days=d)).weekday()] for d in range(30)]
        )[0]

        session_start = pick_session_time(base_date, day_offset)
        session_id    = session_idx + 1

        n_min, n_max = profile["n_events"]
        n_events     = random.randint(n_min, n_max)

        current       = "INICIO"
        event_time_ms = random.randint(500, 3000)
        presence      = 1 if random.random() < profile["presence"] else 0

        for _ in range(n_events):
            dwell = generate_dwell(current, profile)

            rows.append({
                "id":              event_id,
                "session_id":      session_id,
                "event_time_ms":   event_time_ms,
                "screen_id":       current,
                "dwell_ms":        dwell,
                "hora_sessao":     session_start.hour,
                "dia_semana":      session_start.strftime("%A"),
                "timestamp":       (session_start + timedelta(milliseconds=event_time_ms)).strftime("%Y-%m-%d %H:%M:%S"),
                "presence_detect": presence,
                "perfil_usuario":  profile_name,
            })

            event_id      += 1
            event_time_ms += dwell + random.randint(100, 800)
            current        = next_screen(current, profile)

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["id", "session_id", "event_time_ms", "screen_id",
                  "dwell_ms", "hora_sessao", "dia_semana", "timestamp",
                  "presence_detect", "perfil_usuario"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[gera_dados] {len(rows)} eventos em {max(r['session_id'] for r in rows)} sessões → {path}")


if __name__ == "__main__":
    output = Path(__file__).parent / "dados_simulados.csv"
    rows   = generate_sessions(n_sessions=300)
    save_csv(rows, output)
