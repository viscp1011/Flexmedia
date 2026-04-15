"""
recomendacao.py — Sistema de Recomendação via Cadeia de Markov (Sprint 4).

Aprende os padrões de navegação do usuário a partir do histórico de sessões
e recomenda qual tela mostrar a seguir, com a probabilidade de cada transição.

Exemplo de uso:
    from recomendacao import MarkovRecomendador
    rec = MarkovRecomendador()
    rec.fit(df)           # df com colunas session_id, screen_id, event_time_ms
    rec.recomendar("CATALOGO")
    # → {"PROMOCOES": 0.68, "INICIO": 0.22, "CATALOGO": 0.10}
"""
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import CSV_PATH

OUTPUT_DIR = Path(__file__).parent / "graficos"
OUTPUT_DIR.mkdir(exist_ok=True)


class MarkovRecomendador:
    """
    Modelo de Cadeia de Markov de primeira ordem para navegação entre telas.

    Aprende P(próxima_tela | tela_atual) a partir de sequências de sessão
    e usa essas probabilidades para recomendar o próximo conteúdo.
    """

    SCREENS = ["INICIO", "CATALOGO", "PROMOCOES"]

    def __init__(self):
        # Matriz de contagem de transições: counts[origem][destino]
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.probs:  dict[str, dict[str, float]] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "MarkovRecomendador":
        """
        Aprende as probabilidades de transição a partir do DataFrame de eventos.
        Espera colunas: session_id, screen_id, event_time_ms
        """
        df_sorted = df.sort_values(["session_id", "event_time_ms"])

        for session_id, grupo in df_sorted.groupby("session_id"):
            telas = grupo["screen_id"].str.upper().tolist()
            # conta cada par (tela_atual → próxima_tela)
            for i in range(len(telas) - 1):
                origem  = telas[i]
                destino = telas[i + 1]
                self.counts[origem][destino] += 1

        # Normaliza para probabilidades
        for origem, destinos in self.counts.items():
            total = sum(destinos.values())
            self.probs[origem] = {dest: cnt / total for dest, cnt in destinos.items()}

        self.fitted = True
        return self

    def recomendar(self, tela_atual: str) -> dict[str, float]:
        """
        Retorna dicionário {tela: probabilidade} ordenado do mais ao menos provável.
        Se a tela não foi vista no treino, retorna distribuição uniforme.
        """
        tela_atual = tela_atual.upper()
        if not self.fitted:
            raise RuntimeError("Chame fit() antes de recomendar().")

        if tela_atual in self.probs:
            probs = dict(self.probs[tela_atual])
        else:
            # Fallback: uniforme entre as outras telas
            outras = [s for s in self.SCREENS if s != tela_atual]
            probs = {s: 1 / len(outras) for s in outras}

        # Ordena por probabilidade decrescente
        return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    def top1(self, tela_atual: str) -> str:
        """Retorna apenas a tela mais provável a seguir."""
        rec = self.recomendar(tela_atual)
        return max(rec, key=rec.get)

    def matriz_transicao(self) -> pd.DataFrame:
        """Retorna a matriz de transição como DataFrame (linhas=origem, colunas=destino)."""
        screens = self.SCREENS
        data = {}
        for origem in screens:
            row = {}
            for destino in screens:
                row[destino] = self.probs.get(origem, {}).get(destino, 0.0)
            data[origem] = row
        df = pd.DataFrame(data).T  # linhas = origem
        df.index.name   = "Origem"
        df.columns.name = "Destino"
        return df

    def plot_matriz_transicao(self) -> None:
        """Salva heatmap da matriz de transição."""
        mt = self.matriz_transicao()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(mt, annot=True, fmt=".2f", cmap="Blues",
                    linewidths=0.5, ax=ax,
                    vmin=0, vmax=1,
                    cbar_kws={"label": "Probabilidade"})
        ax.set_title("Matriz de Transição — Cadeia de Markov\nP(próxima tela | tela atual)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Próxima Tela")
        ax.set_ylabel("Tela Atual")
        plt.tight_layout()
        path = OUTPUT_DIR / "09_matriz_transicao.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[recomendacao] Salvo: {path.name}")


# ─────────────────────── Main (demonstração) ───────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    rec = MarkovRecomendador()
    rec.fit(df)
    rec.plot_matriz_transicao()

    print("\n════ SISTEMA DE RECOMENDAÇÃO — CADEIA DE MARKOV ════\n")
    print("Matriz de Transição:")
    print(rec.matriz_transicao().to_string(float_format="{:.2%}".format))

    print("\nRecomendações:")
    for tela in ["INICIO", "CATALOGO", "PROMOCOES"]:
        top = rec.top1(tela)
        probs = rec.recomendar(tela)
        probs_str = " | ".join(f"{t}: {p:.0%}" for t, p in probs.items())
        print(f"  {tela:12s} → próxima recomendada: {top:12s} ({probs_str})")
