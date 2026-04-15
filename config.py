"""
config.py — Carrega variáveis de ambiente de forma segura.
Cria um arquivo .env baseado em .env.example com suas credenciais reais.
"""
import os
from pathlib import Path

# Tenta carregar python-dotenv se disponível
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # Se não tiver python-dotenv, usa variáveis de ambiente do sistema

ORACLE_USER     = os.getenv("ORACLE_USER", "rm567895")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "101102")
ORACLE_HOST     = os.getenv("ORACLE_HOST", "oracle.fiap.com.br")
ORACLE_PORT     = int(os.getenv("ORACLE_PORT", "1521"))
ORACLE_SID      = os.getenv("ORACLE_SID", "ORCL")

TABLE_NAME      = "SCREEN_TIME"

# Arquivo de dados simulados (fallback quando Oracle não está disponível)
CSV_PATH        = Path(__file__).parent / "dados_simulados.csv"
LOGS_PATH       = Path(__file__).parent / "logs_telas.txt"
