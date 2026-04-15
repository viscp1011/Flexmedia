"""
importa_logs_oracle.py — Importa logs do Monitor Serial para o Oracle.

Melhorias em relação à Sprint 2:
  - Credenciais carregadas via config.py (nunca hardcoded)
  - Caminho do arquivo via argumento CLI ou variável de ambiente
  - Validação de dados antes de inserir
  - Detecção e descarte de duplicatas (EVENT_TIME_MS + SCREEN_ID)
  - Log de execução com contagem detalhada
"""
import sys
import os
import argparse
from pathlib import Path

try:
    import oracledb
except ImportError:
    print("[ERRO] oracledb não instalado. Execute: pip install oracledb")
    sys.exit(1)

from config import (
    ORACLE_USER, ORACLE_PASSWORD, ORACLE_HOST,
    ORACLE_PORT, ORACLE_SID, TABLE_NAME, LOGS_PATH
)

VALID_SCREENS = {"INICIO", "CATALOGO", "PROMOCOES"}


# ───────────────────────────── Parse ─────────────────────────────

def parse_line(line: str) -> tuple | None:
    """
    Valida e interpreta uma linha do log no formato:
        EVENT_TIME_MS,SCREEN_ID,DWELL_MS
    Retorna (event_time_ms, screen_id, dwell_ms) ou None se inválida.
    """
    line = line.strip()
    if not line or "," not in line:
        return None

    parts = line.split(",")
    if len(parts) != 3:
        return None

    try:
        event_time_ms = int(parts[0])
        screen_id     = parts[1].strip().upper()
        dwell_ms      = int(parts[2])
    except ValueError:
        return None

    # Validações de integridade
    if screen_id not in VALID_SCREENS:
        return None
    if event_time_ms < 0 or dwell_ms < 0:
        return None
    if dwell_ms > 600_000:  # ignora permanências > 10 minutos (provavelmente sensor travado)
        return None

    return event_time_ms, screen_id, dwell_ms


# ───────────────────────────── Oracle ─────────────────────────────

def get_connection():
    dsn = oracledb.makedsn(ORACLE_HOST, ORACLE_PORT, sid=ORACLE_SID)
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=dsn)


def load_existing_keys(cursor) -> set:
    """Carrega pares (EVENT_TIME_MS, SCREEN_ID) já presentes no banco."""
    cursor.execute(f"SELECT EVENT_TIME_MS, SCREEN_ID FROM {TABLE_NAME}")
    return {(int(r[0]), str(r[1]).upper()) for r in cursor.fetchall()}


def import_logs(log_file: Path) -> None:
    if not log_file.exists():
        print(f"[ERRO] Arquivo não encontrado: {log_file}")
        sys.exit(1)

    conn   = get_connection()
    cursor = conn.cursor()

    existing = load_existing_keys(cursor)
    print(f"[info] {len(existing)} registros já existem no banco.")

    total_linhas = inseridas = ignoradas_invalidas = ignoradas_duplicatas = 0

    with open(log_file, encoding="utf-8") as f:
        for raw_line in f:
            total_linhas += 1
            parsed = parse_line(raw_line)

            if parsed is None:
                ignoradas_invalidas += 1
                continue

            event_time_ms, screen_id, dwell_ms = parsed
            key = (event_time_ms, screen_id)

            if key in existing:
                ignoradas_duplicatas += 1
                continue

            cursor.execute(
                f"INSERT INTO {TABLE_NAME} (EVENT_TIME_MS, SCREEN_ID, DWELL_MS) VALUES (:1, :2, :3)",
                (event_time_ms, screen_id, dwell_ms)
            )
            existing.add(key)
            inseridas += 1

    conn.commit()
    cursor.close()
    conn.close()

    print("─" * 50)
    print(f"Linhas lidas         : {total_linhas}")
    print(f"Inválidas ignoradas  : {ignoradas_invalidas}")
    print(f"Duplicatas ignoradas : {ignoradas_duplicatas}")
    print(f"Registros inseridos  : {inseridas}")
    print("─" * 50)


# ───────────────────────────── Main ─────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Importa logs do Totem para Oracle.")
    parser.add_argument(
        "--file", "-f",
        type=Path,
        default=Path(os.getenv("LOG_FILE", str(LOGS_PATH))),
        help="Caminho para o arquivo de logs (.txt)"
    )
    args = parser.parse_args()
    import_logs(args.file)
