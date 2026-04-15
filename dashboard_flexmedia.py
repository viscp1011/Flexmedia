"""
dashboard_flexmedia.py — Dashboard unificado do Totem Flexmedia (Sprints 3 + 4).

5 abas:
  📊 Visão Geral      — KPIs globais, amostra dos dados, distribuição de dwell
  📈 Análise          — Padrões temporais, por tela, heatmap hora×tela (Sprint 3)
  🤖 Machine Learning — Engajamento + predição de próxima tela (Sprint 3)
  🔮 Recomendação     — Cadeia de Markov, matriz de transição (Sprint 4)
  💬 Chat IA          — Interface conversacional sobre os dados do totem (Sprint 4)

Fonte de dados: dados_simulados.csv (fallback) ou Oracle DB
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importa módulos locais
sys.path.insert(0, str(Path(__file__).parent))
from config import CSV_PATH
from recomendacao import MarkovRecomendador

# ─────────────────────── Configuração da página ───────────────────────

st.set_page_config(
    page_title="Totem Flexmedia – Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── Carregamento de dados ───────────────────────

def _load_oracle() -> pd.DataFrame | None:
    """Tenta carregar dados do Oracle. Retorna None se falhar."""
    try:
        import oracledb
        from config import ORACLE_USER, ORACLE_PASSWORD, ORACLE_HOST, ORACLE_PORT, ORACLE_SID, TABLE_NAME
        dsn = oracledb.makedsn(ORACLE_HOST, ORACLE_PORT, sid=ORACLE_SID)
        with oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=dsn) as conn:
            df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
            df.columns = [c.lower() for c in df.columns]
            if "session_id" not in df.columns:
                df["session_id"] = 1
            if "hora_sessao" not in df.columns:
                df["hora_sessao"] = 12
            if "dia_semana" not in df.columns:
                df["dia_semana"] = "Monday"
            if "presence_detect" not in df.columns:
                df["presence_detect"] = 1
            if "perfil_usuario" not in df.columns:
                df["perfil_usuario"] = "desconhecido"
            return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_data(usar_oracle: bool = True) -> pd.DataFrame:
    """Carrega dados: Oracle se disponível e selecionado, senão CSV simulado."""
    if usar_oracle:
        df = _load_oracle()
        if df is not None:
            return df

    # CSV simulado
    if not CSV_PATH.exists():
        st.error("Dados não encontrados. Execute gera_dados.py primeiro.")
        st.stop()
    return pd.read_csv(CSV_PATH)


@st.cache_data(show_spinner=False)
def prepare_data(df_raw: pd.DataFrame):
    """Feature engineering compartilhado entre abas."""
    df = df_raw.copy()
    df["screen_id"] = df["screen_id"].str.upper()
    df["dwell_sec"] = df["dwell_ms"] / 1000.0

    mediana = df["dwell_ms"].median()
    df["engajamento"] = df["dwell_ms"].apply(lambda x: "ALTO" if x >= mediana else "BAIXO")

    df = df.sort_values(["session_id", "event_time_ms"])
    df["pos_na_sessao"] = df.groupby("session_id").cumcount()

    le = LabelEncoder()
    df["screen_enc"] = le.fit_transform(df["screen_id"])

    dias = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["dia_enc"] = df["dia_semana"].map(dias).fillna(0).astype(int)

    df["proxima_tela"] = df.groupby("session_id")["screen_id"].shift(-1)

    return df, mediana


@st.cache_data(show_spinner=False)
def treinar_modelos(df: pd.DataFrame):
    """Treina Random Forest para engajamento e próxima tela."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    # ── Modelo 1: Engajamento ──
    features = ["dwell_ms", "hora_sessao", "screen_enc", "pos_na_sessao", "dia_enc"]
    y_eng = (df["engajamento"] == "ALTO").astype(int)
    X = df[features]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_eng, test_size=0.25, random_state=42)
    rf_eng = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_eng.fit(X_tr, y_tr)
    y_pred_eng = rf_eng.predict(X_te)
    cv_eng = cross_val_score(rf_eng, X, y_eng, cv=5).mean()

    metricas_eng = {
        "acuracia":  accuracy_score(y_te, y_pred_eng),
        "precisao":  precision_score(y_te, y_pred_eng, zero_division=0),
        "recall":    recall_score(y_te, y_pred_eng, zero_division=0),
        "f1":        f1_score(y_te, y_pred_eng, zero_division=0),
        "cv_media":  cv_eng,
        "conf_mat":  confusion_matrix(y_te, y_pred_eng),
        "importances": rf_eng.feature_importances_,
        "features":  features,
    }

    # ── Modelo 2: Próxima Tela ──
    df2 = df.dropna(subset=["proxima_tela"]).copy()
    le_prox = LabelEncoder()
    df2["prox_enc"] = le_prox.fit_transform(df2["proxima_tela"])

    X2 = df2[features]
    y2 = df2["prox_enc"]
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.25, random_state=42, stratify=y2)
    rf_prox = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_prox.fit(X2_tr, y2_tr)
    y2_pred = rf_prox.predict(X2_te)
    cv_prox = cross_val_score(rf_prox, X2, y2, cv=5).mean()

    metricas_prox = {
        "acuracia":  accuracy_score(y2_te, y2_pred),
        "precisao":  precision_score(y2_te, y2_pred, average="macro", zero_division=0),
        "recall":    recall_score(y2_te, y2_pred, average="macro", zero_division=0),
        "f1":        f1_score(y2_te, y2_pred, average="macro", zero_division=0),
        "cv_media":  cv_prox,
        "conf_mat":  confusion_matrix(y2_te, y2_pred),
        "classes":   le_prox.classes_,
        "importances": rf_prox.feature_importances_,
        "features":  features,
        "model":     rf_prox,
        "le_prox":   le_prox,
        "X":         X2,
    }

    return metricas_eng, metricas_prox, rf_eng


# ─────────────────────── Sidebar ───────────────────────

with st.sidebar:
    st.image("https://placehold.co/200x60/1a1a2e/4C9BE8?text=FLEXMEDIA&font=raleway", use_container_width=True)
    st.markdown("### 🛍️ Totem Inteligente")
    st.caption("Challenge Flexmedia — FIAP | Sprints 3 & 4")
    st.divider()

    usar_oracle = st.toggle("☁️ Usar Oracle DB", value=True,
                            help="Ativado: lê da tabela Oracle FIAP. Desativado: usa CSV simulado (2805 eventos).")

    with st.spinner("Carregando dados..."):
        df_raw = load_data(usar_oracle=usar_oracle)
        df, mediana = prepare_data(df_raw)

    # Detecta fonte real: CSV simulado tem coluna perfil_usuario preenchida
    tem_perfil = "perfil_usuario" in df.columns and (df["perfil_usuario"] != "desconhecido").any()
    fonte = "📄 CSV Simulado" if tem_perfil else "☁️ Oracle DB"
    st.success(f"Fonte: {fonte}")
    st.metric("Total de eventos", len(df))
    st.metric("Sessões únicas", df["session_id"].nunique())
    st.metric("Telas monitoradas", df["screen_id"].nunique())
    st.divider()

    tela_filtro = st.multiselect(
        "Filtrar telas:",
        options=["INICIO", "CATALOGO", "PROMOCOES"],
        default=["INICIO", "CATALOGO", "PROMOCOES"],
    )
    df_filtrado = df[df["screen_id"].isin(tela_filtro)] if tela_filtro else df

# ─────────────────────── Abas ───────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral",
    "📈 Análise Estatística",
    "🤖 Machine Learning",
    "🔮 Recomendação IA",
    "💬 Chat IA",
])

CORES = {"INICIO": "#4C9BE8", "CATALOGO": "#F5A623", "PROMOCOES": "#7ED321"}

# ══════════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Visão Geral das Interações")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de eventos", len(df_filtrado))
    c2.metric("Dwell médio global", f"{df_filtrado['dwell_sec'].mean():.1f} s")
    c3.metric("Dwell mediano", f"{df_filtrado['dwell_ms'].median()/1000:.1f} s")
    c4.metric("Hora de pico", f"{df_filtrado.groupby('hora_sessao').size().idxmax()}h")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribuição de Permanência (dwell)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for tela in ["INICIO", "CATALOGO", "PROMOCOES"]:
            subset = df_filtrado[df_filtrado["screen_id"] == tela]["dwell_sec"]
            if not subset.empty:
                ax.hist(subset, bins=20, alpha=0.6, label=tela, color=CORES.get(tela, "gray"))
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência")
        ax.legend()
        ax.set_title("Histograma de Dwell por Tela")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Engajamento por Tela")
        eng_pivot = (df_filtrado.groupby(["screen_id", "engajamento"])
                                .size().unstack(fill_value=0))
        for col in ["ALTO", "BAIXO"]:
            if col not in eng_pivot.columns:
                eng_pivot[col] = 0
        fig, ax = plt.subplots(figsize=(6, 3.5))
        eng_pivot[["ALTO", "BAIXO"]].plot(kind="bar", stacked=True, ax=ax,
                                           color=["#7ED321", "#E74C3C"],
                                           edgecolor="white", rot=0)
        ax.set_title("ALTO vs BAIXO engajamento por tela")
        ax.set_xlabel("")
        ax.legend(title="Engajamento")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Amostra dos dados")
    st.dataframe(df_filtrado.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ABA 2 — ANÁLISE ESTATÍSTICA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("📈 Análise Estatística — Padrões de Uso")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Acessos por Tela")
        contagem = df_filtrado.groupby("screen_id").size().reindex(["INICIO","CATALOGO","PROMOCOES"]).dropna()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(contagem.index, contagem.values,
                      color=[CORES.get(s, "gray") for s in contagem.index],
                      edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_ylabel("Nº de acessos")
        ax.set_ylim(0, contagem.max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Tempo Médio por Tela")
        media = df_filtrado.groupby("screen_id")["dwell_sec"].mean().reindex(["INICIO","CATALOGO","PROMOCOES"]).dropna()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(media.index, media.values,
                      color=[CORES.get(s, "gray") for s in media.index],
                      edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.1f s", padding=3)
        ax.set_ylabel("Tempo médio (s)")
        ax.set_ylim(0, media.max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Padrão Temporal — Acessos por Hora do Dia")
    por_hora = df_filtrado.groupby("hora_sessao").size().reset_index(name="acessos")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(por_hora["hora_sessao"], por_hora["acessos"],
            marker="o", linewidth=2, color="#4C9BE8", markersize=5)
    ax.fill_between(por_hora["hora_sessao"], por_hora["acessos"], alpha=0.12, color="#4C9BE8")
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Acessos")
    ax.set_xticks(range(0, 24))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Heatmap: Hora × Tela")
    pivot = df_filtrado.pivot_table(index="hora_sessao", columns="screen_id",
                                    values="dwell_ms", aggfunc="count", fill_value=0)
    for col in ["INICIO", "CATALOGO", "PROMOCOES"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["INICIO", "CATALOGO", "PROMOCOES"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Nº de acessos"})
    ax.set_xlabel("Tela")
    ax.set_ylabel("Hora do dia")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Boxplot de Permanência por Tela")
    fig, ax = plt.subplots(figsize=(7, 4))
    order = [s for s in ["INICIO", "CATALOGO", "PROMOCOES"] if s in df_filtrado["screen_id"].values]
    data_box = [df_filtrado[df_filtrado["screen_id"] == s]["dwell_sec"].values for s in order]
    bp = ax.boxplot(data_box, labels=order, patch_artist=True)
    for patch, tela in zip(bp["boxes"], order):
        patch.set_facecolor(CORES.get(tela, "gray"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Distribuição de permanência por tela")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# ABA 3 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Modelos de Machine Learning")

    with st.spinner("Treinando modelos..."):
        met_eng, met_prox, rf_eng = treinar_modelos(df)

    # ── Modelo 1: Engajamento ──
    st.subheader("Modelo 1 — Classificador de Engajamento (Random Forest)")
    st.markdown(
        "Classifica cada acesso como **ALTO** ou **BAIXO** engajamento "
        "usando 5 features: tempo de permanência, hora da sessão, tela atual, "
        "posição na sessão e dia da semana."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Acurácia",  f"{met_eng['acuracia']:.2%}")
    m2.metric("Precisão",  f"{met_eng['precisao']:.2%}")
    m3.metric("Recall",    f"{met_eng['recall']:.2%}")
    m4.metric("F1-Score",  f"{met_eng['f1']:.2%}")
    m5.metric("CV 5-fold", f"{met_eng['cv_media']:.2%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Feature Importance**")
        fi_df = pd.DataFrame({"Feature": met_eng["features"],
                               "Importância": met_eng["importances"]}).sort_values("Importância")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(fi_df["Feature"], fi_df["Importância"], color="#4C9BE8")
        ax.set_xlabel("Importância")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Matriz de Confusão**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(met_eng["conf_mat"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["BAIXO","ALTO"], yticklabels=["BAIXO","ALTO"], ax=ax)
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()

    # ── Modelo 2: Próxima Tela ──
    st.subheader("Modelo 2 — Preditor de Próxima Tela (Random Forest Multiclass)")
    st.markdown("Prevê qual tela o usuário irá visitar a seguir, com base no comportamento atual.")

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Acurácia",  f"{met_prox['acuracia']:.2%}")
    p2.metric("Precisão",  f"{met_prox['precisao']:.2%}")
    p3.metric("Recall",    f"{met_prox['recall']:.2%}")
    p4.metric("F1-Score",  f"{met_prox['f1']:.2%}")
    p5.metric("CV 5-fold", f"{met_prox['cv_media']:.2%}")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Feature Importance**")
        fi_df2 = pd.DataFrame({"Feature": met_prox["features"],
                                "Importância": met_prox["importances"]}).sort_values("Importância")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(fi_df2["Feature"], fi_df2["Importância"], color="#F5A623")
        ax.set_xlabel("Importância")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown("**Matriz de Confusão**")
        classes = list(met_prox["classes"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(met_prox["conf_mat"], annot=True, fmt="d", cmap="Oranges",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()

    # ── Teste interativo ──
    st.subheader("🔬 Teste o Modelo de Engajamento")
    col_i, col_j = st.columns(2)
    with col_i:
        dwell_in = st.slider("Tempo de permanência (ms):",
                             min_value=200, max_value=15000, value=5000, step=100)
        hora_in  = st.slider("Hora do acesso:", 0, 23, 14)
    with col_j:
        screen_in = st.selectbox("Tela:", ["INICIO", "CATALOGO", "PROMOCOES"])
        pos_in    = st.slider("Posição na sessão:", 0, 10, 2)

    screen_enc_map = {"CATALOGO": 0, "INICIO": 1, "PROMOCOES": 2}
    X_input = pd.DataFrame([[dwell_in, hora_in, screen_enc_map.get(screen_in, 1), pos_in, 0]],
                            columns=["dwell_ms","hora_sessao","screen_enc","pos_na_sessao","dia_enc"])
    pred_label = "ALTO" if rf_eng.predict(X_input)[0] == 1 else "BAIXO"
    pred_proba = rf_eng.predict_proba(X_input)[0]
    confianca  = max(pred_proba)

    if pred_label == "ALTO":
        st.success(f"✅ Engajamento previsto: **ALTO** (confiança: {confianca:.0%})")
    else:
        st.warning(f"⚠️ Engajamento previsto: **BAIXO** (confiança: {confianca:.0%})")


# ══════════════════════════════════════════════════════════════
# ABA 4 — RECOMENDAÇÃO IA
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("🔮 Sistema de Recomendação — Cadeia de Markov")
    st.markdown(
        "Aprende os **padrões de navegação** dos usuários entre telas "
        "e recomenda qual conteúdo exibir a seguir para maximizar o engajamento."
    )

    rec = MarkovRecomendador()
    rec.fit(df)
    mt = rec.matriz_transicao()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Matriz de Transição")
        st.markdown("Probabilidade de ir da tela **origem** para cada tela **destino**:")
        # Exibe a matriz formatada em %
        mt_pct = mt.applymap(lambda x: f"{x:.1%}")
        st.dataframe(mt_pct, use_container_width=True)

    with col_b:
        st.subheader("Heatmap de Transições")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(mt, annot=True, fmt=".2f", cmap="Blues",
                    linewidths=0.5, ax=ax, vmin=0, vmax=1,
                    cbar_kws={"label": "Probabilidade"})
        ax.set_xlabel("Próxima tela")
        ax.set_ylabel("Tela atual")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()

    st.subheader("🎯 Recomendação em Tempo Real")
    tela_atual = st.selectbox("Usuário está na tela:", ["INICIO", "CATALOGO", "PROMOCOES"], key="rec_tela")
    probs = rec.recomendar(tela_atual)
    top = rec.top1(tela_atual)

    st.success(f"✅ Recomendação: mostrar **{top}** a seguir")

    st.markdown("**Probabilidades de transição:**")
    for tela, prob in probs.items():
        cor = "🟢" if tela == top else "⚪"
        st.progress(prob, text=f"{cor} {tela}: {prob:.1%}")

    st.divider()

    st.subheader("📊 Análise das Sequências de Navegação mais Frequentes")
    df_seq = df.sort_values(["session_id", "event_time_ms"])
    df_seq["proxima"] = df_seq.groupby("session_id")["screen_id"].shift(-1)
    df_seq2 = df_seq.dropna(subset=["proxima"])
    seq_count = (df_seq2.groupby(["screen_id", "proxima"])
                         .size().reset_index(name="count")
                         .sort_values("count", ascending=False)
                         .head(9))
    seq_count["sequencia"] = seq_count["screen_id"] + " → " + seq_count["proxima"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(seq_count["sequencia"], seq_count["count"], color="#4C9BE8", edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Frequência")
    ax.set_title("Top sequências de navegação observadas")
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# ABA 5 — CHAT IA
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("💬 Chat IA — Assistente do Totem")
    st.markdown(
        "Faça perguntas sobre os dados de uso do totem. "
        "O assistente interpreta os dados e responde em linguagem natural."
    )

    # ── Base de conhecimento calculada ──
    total_eventos = len(df)
    total_sessoes = df["session_id"].nunique()
    tela_mais     = df.groupby("screen_id").size().idxmax()
    tela_menos    = df.groupby("screen_id").size().idxmin()
    hora_pico     = df.groupby("hora_sessao").size().idxmax()
    dwell_medio   = df["dwell_sec"].mean()
    tela_maior_dwell = df.groupby("screen_id")["dwell_sec"].mean().idxmax()
    tela_menor_dwell = df.groupby("screen_id")["dwell_sec"].mean().idxmin()
    pct_alto      = (df["engajamento"] == "ALTO").mean()
    rec_ini       = rec.top1("INICIO")
    rec_cat       = rec.top1("CATALOGO")
    rec_pro       = rec.top1("PROMOCOES")

    def responder(pergunta: str) -> str:
        """
        Responde perguntas sobre os dados do totem usando regras e os dados calculados.
        Reconhece palavras-chave em português.
        """
        p = pergunta.lower()

        # Engajamento
        if any(w in p for w in ["engajamento", "engajado", "engajar"]):
            return (
                f"O engajamento geral é de **{pct_alto:.0%} ALTO** e {1-pct_alto:.0%} BAIXO. "
                f"A tela com maior tempo médio de permanência é **{tela_maior_dwell}** "
                f"({df.groupby('screen_id')['dwell_sec'].mean()[tela_maior_dwell]:.1f}s), "
                f"indicando que os usuários se engajam mais nessa tela."
            )

        # Tela mais acessada
        if any(w in p for w in ["mais acessada", "mais visitada", "mais popular", "mais usada"]):
            qtd = df.groupby("screen_id").size()[tela_mais]
            return f"A tela mais acessada é **{tela_mais}**, com **{qtd} acessos** registrados."

        # Tela menos acessada
        if any(w in p for w in ["menos acessada", "menos visitada", "menos popular"]):
            qtd = df.groupby("screen_id").size()[tela_menos]
            return f"A tela menos acessada é **{tela_menos}**, com apenas **{qtd} acessos**."

        # Tempo / permanência
        if any(w in p for w in ["tempo", "permanência", "dwell", "quanto tempo", "ficam"]):
            return (
                f"O tempo médio global de permanência nas telas é de **{dwell_medio:.1f} segundos**. "
                f"A tela com maior permanência média é **{tela_maior_dwell}** "
                f"({df.groupby('screen_id')['dwell_sec'].mean()[tela_maior_dwell]:.1f}s) e a menor é "
                f"**{tela_menor_dwell}** ({df.groupby('screen_id')['dwell_sec'].mean()[tela_menor_dwell]:.1f}s)."
            )

        # Hora de pico
        if any(w in p for w in ["hora", "pico", "horário", "quando", "período"]):
            qtd_pico = df.groupby("hora_sessao").size()[hora_pico]
            return (
                f"O horário de pico é às **{hora_pico}h**, com **{qtd_pico} acessos** registrados. "
                f"Os períodos de maior movimento são manhã (9–12h), tarde (14–18h) e noite (19–21h)."
            )

        # Sessões / usuários
        if any(w in p for w in ["sessão", "sessões", "usuário", "usuários", "visita"]):
            avg_ev = total_eventos / total_sessoes
            return (
                f"Foram registradas **{total_sessoes} sessões** únicas, com um total de "
                f"**{total_eventos} eventos**. Em média, cada sessão gera **{avg_ev:.1f} trocas de tela**."
            )

        # Recomendação
        if any(w in p for w in ["recomendar", "recomendação", "próxima", "sugestão", "mostrar"]):
            return (
                f"Com base nos padrões de navegação, a recomendação é:\n\n"
                f"- Usuário em **INÍCIO** → mostrar **{rec_ini}**\n"
                f"- Usuário em **CATÁLOGO** → mostrar **{rec_cat}**\n"
                f"- Usuário em **PROMOÇÕES** → mostrar **{rec_pro}**"
            )

        # Total de eventos
        if any(w in p for w in ["total", "quantos", "quantas", "eventos", "registros"]):
            return f"O sistema registrou **{total_eventos} eventos** de interação ao longo de **{total_sessoes} sessões**."

        # Catálogo
        if "catálogo" in p or "catalogo" in p:
            qtd = df[df["screen_id"] == "CATALOGO"].shape[0]
            media = df[df["screen_id"] == "CATALOGO"]["dwell_sec"].mean()
            return f"A tela **CATÁLOGO** teve **{qtd} acessos** com tempo médio de **{media:.1f}s** por visita."

        # Promoções
        if "promoç" in p or "promocao" in p or "promoção" in p:
            qtd = df[df["screen_id"] == "PROMOCOES"].shape[0]
            media = df[df["screen_id"] == "PROMOCOES"]["dwell_sec"].mean()
            return f"A tela **PROMOÇÕES** teve **{qtd} acessos** com tempo médio de **{media:.1f}s** por visita."

        # Início
        if "início" in p or "inicio" in p:
            qtd = df[df["screen_id"] == "INICIO"].shape[0]
            media = df[df["screen_id"] == "INICIO"]["dwell_sec"].mean()
            return f"A tela **INÍCIO** teve **{qtd} acessos** com tempo médio de **{media:.1f}s** por visita."

        # Fallback
        return (
            "Não entendi completamente sua pergunta. Experimente perguntar sobre:\n\n"
            "- *Qual tela é mais acessada?*\n"
            "- *Qual o horário de pico?*\n"
            "- *Quanto tempo os usuários ficam no catálogo?*\n"
            "- *Como está o engajamento?*\n"
            "- *Quantas sessões foram registradas?*\n"
            "- *O que recomendar para um usuário no início?*"
        )

    # ── Histórico de chat ──
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = [
            {"role": "assistant", "content":
             "Olá! Sou o assistente do Totem Flexmedia. Posso responder perguntas "
             "sobre os dados de uso do totem — acessos, engajamento, horários e recomendações. "
             "O que você gostaria de saber?"}
        ]

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre os dados do totem..."):
        st.session_state.mensagens.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        resposta = responder(prompt)
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)

    # Sugestões rápidas
    st.markdown("**💡 Sugestões de perguntas:**")
    sugestoes = [
        "Qual tela é mais acessada?",
        "Qual o horário de pico?",
        "Como está o engajamento?",
        "Quanto tempo ficam no catálogo?",
        "O que recomendar para o início?",
    ]
    cols = st.columns(len(sugestoes))
    for i, sug in enumerate(sugestoes):
        if cols[i].button(sug, key=f"sug_{i}"):
            resposta = responder(sug)
            st.session_state.mensagens.append({"role": "user", "content": sug})
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
            st.rerun()
