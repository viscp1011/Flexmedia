"""
analise_estatistica.py — Análise estatística do Totem Flexmedia (Sprint 3).

Gera gráficos salvos em /graficos/:
  1. Distribuição de dwell_ms por tela (boxplot)
  2. Quantidade de acessos por tela (barplot)
  3. Tempo médio por tela (barplot)
  4. Padrão temporal: acessos por hora do dia (lineplot)
  5. Heatmap: hora × tela (quantidade de acessos)
  6. Distribuição de engajamento por tela (stacked bar)
  7. Distribuição de perfis de usuário (pie + bar)

Usa: dados_simulados.csv (fallback quando Oracle indisponível)
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sem display — compatível com servidor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import CSV_PATH

OUTPUT_DIR = Path(__file__).parent / "graficos"
OUTPUT_DIR.mkdir(exist_ok=True)

# Paleta de cores por tela
CORES = {
    "INICIO":    "#4C9BE8",
    "CATALOGO":  "#F5A623",
    "PROMOCOES": "#7ED321",
}


# ─────────────────────── Carregamento ───────────────────────

def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print("[analise] CSV não encontrado. Execute gera_dados.py primeiro.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df["screen_id"] = df["screen_id"].str.upper()
    df["dwell_sec"] = df["dwell_ms"] / 1000.0

    # Engajamento: ALTO se dwell >= mediana global
    mediana = df["dwell_ms"].median()
    df["engajamento"] = df["dwell_ms"].apply(lambda x: "ALTO" if x >= mediana else "BAIXO")

    return df


# ─────────────────────── Gráficos ───────────────────────

def plot_boxplot_dwell(df: pd.DataFrame) -> None:
    """Boxplot de dwell_sec por tela — mostra distribuição e outliers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["INICIO", "CATALOGO", "PROMOCOES"]
    palette = [CORES[s] for s in order]
    sns.boxplot(data=df, x="screen_id", y="dwell_sec",
                order=order, palette=palette, ax=ax, width=0.5)
    ax.set_title("Distribuição de Tempo de Permanência por Tela", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tela")
    ax.set_ylabel("Tempo (segundos)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f s"))
    plt.tight_layout()
    path = OUTPUT_DIR / "01_boxplot_dwell.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def plot_acessos_por_tela(df: pd.DataFrame) -> None:
    """Barplot de quantidade de acessos por tela."""
    contagem = df.groupby("screen_id").size().reindex(["INICIO", "CATALOGO", "PROMOCOES"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(contagem.index, contagem.values,
                  color=[CORES[s] for s in contagem.index], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%d", fontsize=10, padding=3)
    ax.set_title("Quantidade de Acessos por Tela", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tela")
    ax.set_ylabel("Nº de Acessos")
    ax.set_ylim(0, contagem.max() * 1.15)
    plt.tight_layout()
    path = OUTPUT_DIR / "02_acessos_por_tela.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def plot_tempo_medio(df: pd.DataFrame) -> None:
    """Barplot de tempo médio de permanência por tela."""
    media = df.groupby("screen_id")["dwell_sec"].mean().reindex(["INICIO", "CATALOGO", "PROMOCOES"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(media.index, media.values,
                  color=[CORES[s] for s in media.index], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.1f s", fontsize=10, padding=3)
    ax.set_title("Tempo Médio de Permanência por Tela", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tela")
    ax.set_ylabel("Tempo Médio (s)")
    ax.set_ylim(0, media.max() * 1.2)
    plt.tight_layout()
    path = OUTPUT_DIR / "03_tempo_medio_tela.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def plot_acessos_por_hora(df: pd.DataFrame) -> None:
    """Lineplot de acessos por hora do dia — revela padrão temporal."""
    por_hora = df.groupby("hora_sessao").size().reset_index(name="acessos")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(por_hora["hora_sessao"], por_hora["acessos"],
            marker="o", linewidth=2, color="#4C9BE8", markersize=6)
    ax.fill_between(por_hora["hora_sessao"], por_hora["acessos"], alpha=0.15, color="#4C9BE8")
    ax.set_title("Acessos ao Totem por Hora do Dia", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora do Dia")
    ax.set_ylabel("Total de Acessos")
    ax.set_xticks(range(0, 24))
    ax.set_xlim(0, 23)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = OUTPUT_DIR / "04_acessos_por_hora.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def plot_heatmap_hora_tela(df: pd.DataFrame) -> None:
    """Heatmap de hora × tela — quantidade de acessos."""
    pivot = df.pivot_table(index="hora_sessao", columns="screen_id",
                           values="id", aggfunc="count", fill_value=0)
    # garante ordem das colunas
    for col in ["INICIO", "CATALOGO", "PROMOCOES"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["INICIO", "CATALOGO", "PROMOCOES"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Nº de Acessos"})
    ax.set_title("Heatmap: Hora do Dia × Tela", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tela")
    ax.set_ylabel("Hora do Dia")
    plt.tight_layout()
    path = OUTPUT_DIR / "05_heatmap_hora_tela.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def plot_engajamento_por_tela(df: pd.DataFrame) -> None:
    """Stacked bar: proporção de engajamento ALTO/BAIXO por tela."""
    eng = (df.groupby(["screen_id", "engajamento"])
             .size()
             .unstack(fill_value=0)
             .reindex(["INICIO", "CATALOGO", "PROMOCOES"]))

    for col in ["ALTO", "BAIXO"]:
        if col not in eng.columns:
            eng[col] = 0
    eng = eng[["ALTO", "BAIXO"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    eng.plot(kind="bar", stacked=True, ax=ax,
             color=["#7ED321", "#E74C3C"], edgecolor="white", rot=0)
    ax.set_title("Engajamento por Tela (ALTO vs BAIXO)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tela")
    ax.set_ylabel("Nº de Acessos")
    ax.legend(title="Engajamento")
    plt.tight_layout()
    path = OUTPUT_DIR / "06_engajamento_por_tela.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def plot_perfis_usuario(df: pd.DataFrame) -> None:
    """Pie + bar mostrando distribuição de sessões por perfil de usuário."""
    if "perfil_usuario" not in df.columns:
        print("[analise] Coluna 'perfil_usuario' não encontrada — pulando gráfico.")
        return

    # Conta sessões únicas por perfil (não eventos)
    perfis = (df.drop_duplicates("session_id")
                .groupby("perfil_usuario")
                .size()
                .sort_values(ascending=False))

    CORES_PERFIL = {
        "engajado":   "#7ED321",
        "curioso":    "#4C9BE8",
        "passante":   "#F5A623",
        "comparador": "#9B59B6",
    }
    cores = [CORES_PERFIL.get(p, "#AAAAAA") for p in perfis.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        perfis.values, labels=perfis.index, colors=cores,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax1.set_title("Distribuição de Perfis de Usuário\n(por sessão)", fontsize=12, fontweight="bold")

    # Bar chart com contagem absoluta
    bars = ax2.bar(perfis.index, perfis.values, color=cores, edgecolor="white", width=0.55)
    ax2.bar_label(bars, fmt="%d sessões", fontsize=10, padding=4)
    ax2.set_title("Sessões por Perfil de Usuário", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Perfil")
    ax2.set_ylabel("Nº de Sessões")
    ax2.set_ylim(0, perfis.max() * 1.18)

    plt.suptitle("Análise de Perfis Comportamentais — Totem Flexmedia",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "07_perfis_usuario.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analise] Salvo: {path.name}")


def print_resumo(df: pd.DataFrame) -> None:
    print("\n" + "═" * 55)
    print("  RESUMO ESTATÍSTICO — TOTEM FLEXMEDIA")
    print("═" * 55)
    print(f"  Total de eventos       : {len(df)}")
    print(f"  Total de sessões       : {df['session_id'].nunique()}")
    print(f"  Telas distintas        : {df['screen_id'].nunique()}")
    print(f"  Dwell médio global     : {df['dwell_sec'].mean():.2f} s")
    print(f"  Dwell mediano global   : {(df['dwell_ms'].median()/1000):.2f} s")
    print(f"  Dwell máximo           : {df['dwell_sec'].max():.2f} s")
    print(f"  Hora de pico           : {df.groupby('hora_sessao').size().idxmax()}h")
    print("─" * 55)
    tbl = (df.groupby("screen_id")
             .agg(acessos=("id","count"),
                  media_seg=("dwell_sec","mean"),
                  total_min=("dwell_sec", lambda x: x.sum()/60))
             .reindex(["INICIO","CATALOGO","PROMOCOES"]))
    print(tbl.to_string())
    print("═" * 55 + "\n")


# ─────────────────────── Main ───────────────────────

if __name__ == "__main__":
    df = load_data()
    print_resumo(df)
    plot_boxplot_dwell(df)
    plot_acessos_por_tela(df)
    plot_tempo_medio(df)
    plot_acessos_por_hora(df)
    plot_heatmap_hora_tela(df)
    plot_engajamento_por_tela(df)
    plot_perfis_usuario(df)
    print(f"\n[analise] Todos os gráficos salvos em: {OUTPUT_DIR}/")
