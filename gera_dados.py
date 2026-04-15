"""
gera_dados.py — Gera um dataset simulado realista para o Totem Flexmedia.

Simula ~30 sessões de uso com padrões coerentes:
- Horários de pico (manhã/tarde/noite)
- Comportamentos distintos por tela (INICIO tem dwell curto, CATALOGO longo)
- Variação de engajamento entre sessões
- Coluna de presença simulada (detecção de pessoa via câmera)

Gera: dados_simulados.csv
"""
import random
import csv
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

SCREENS = ["INICIO", "CATALOGO", "PROMOCOES"]

# Probabilidades de transição entre telas (matriz de transição)
# Representa comportamento real: do INICIO vai mais pra CATALOGO; do CATALOGO vai pra PROMOCOES
TRANSITION = {
    "INICIO":    {"INICIO": 0.05, "CATALOGO": 0.65, "PROMOCOES": 0.30},
    "CATALOGO":  {"INICIO": 0.20, "CATALOGO": 0.10, "PROMOCOES": 0.70},
    "PROMOCOES": {"INICIO": 0.30, "CATALOGO": 0.60, "PROMOCOES": 0.10},
}

# Distribuição de dwell_ms por tela (média, desvio em ms)
DWELL_PARAMS = {
    "INICIO":    (3000, 1500),
    "CATALOGO":  (8000, 3000),
    "PROMOCOES": (5000, 2000),
}

# Horários de pico: manhã (9-12h), tarde (14-18h), noite (19-21h)
PEAK_HOURS = list(range(9, 12)) + list(range(14, 18)) + list(range(19, 22))


def next_screen(current: str) -> str:
    """Escolhe a próxima tela com base na matriz de transição."""
    probs = TRANSITION[current]
    screens = list(probs.keys())
    weights = list(probs.values())
    return random.choices(screens, weights=weights)[0]


def generate_dwell(screen: str) -> int:
    """Gera tempo de permanência com distribuição normal, mínimo 200ms."""
    mean, std = DWELL_PARAMS[screen]
    return max(200, int(random.gauss(mean, std)))


def generate_sessions(n_sessions: int = 35) -> list[dict]:
    """
    Gera n_sessions sessões de uso, cada uma com 4-12 eventos de tela.
    Retorna lista de dicts com colunas do CSV.
    """
    rows = []
    event_id = 1
    # Distribui sessões ao longo de 7 dias
    base_date = datetime(2025, 11, 20, 0, 0, 0)

    for session_id in range(1, n_sessions + 1):
        # Escolhe dia e hora da sessão
        day_offset = random.randint(0, 6)
        hour = random.choice(PEAK_HOURS) if random.random() < 0.75 else random.randint(8, 22)
        minute = random.randint(0, 59)
        session_start = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        # Número de eventos nesta sessão (4 a 12 trocas de tela)
        n_events = random.randint(4, 12)

        current = "INICIO"
        event_time_ms = random.randint(1000, 5000)  # tempo inicial acumulado no ESP32

        # Presença detectada (câmera simulada): 1 se há pessoa, 0 se ausente
        presence = 1 if random.random() < 0.92 else 0

        for _ in range(n_events):
            dwell = generate_dwell(current)

            rows.append({
                "id":              event_id,
                "session_id":      session_id,
                "event_time_ms":   event_time_ms,
                "screen_id":       current,
                "dwell_ms":        dwell,
                "hora_sessao":     session_start.hour,
                "dia_semana":      session_start.strftime("%A"),  # Monday, Tuesday...
                "timestamp":       (session_start + timedelta(milliseconds=event_time_ms)).strftime("%Y-%m-%d %H:%M:%S"),
                "presence_detect": presence,
            })

            event_id += 1
            event_time_ms += dwell + random.randint(100, 500)  # pequena pausa entre eventos
            current = next_screen(current)

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["id", "session_id", "event_time_ms", "screen_id",
                  "dwell_ms", "hora_sessao", "dia_semana", "timestamp", "presence_detect"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[gera_dados] {len(rows)} eventos gerados → {path}")


if __name__ == "__main__":
    output = Path(__file__).parent / "dados_simulados.csv"
    rows = generate_sessions(n_sessions=35)
    save_csv(rows, output)
