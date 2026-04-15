"""
modelo_ml.py — Modelos de Machine Learning para o Totem Flexmedia (Sprint 3).

Modelos treinados:
  1. Classificador de Engajamento (Random Forest)
     Target: ALTO / BAIXO  com base em múltiplas features
     Features: dwell_ms, hora_sessao, screen_enc, pos_na_sessao, dia_enc, perfil_enc

  2. Preditor de Próxima Tela (Random Forest Multiclass)
     Target: qual será a próxima tela visitada
     Features: screen_enc, dwell_ms, hora_sessao, pos_na_sessao, dia_enc, perfil_enc

Saídas:
  - Métricas completas no terminal (acurácia, precisão, recall, F1, matriz de confusão)
  - Gráficos em /graficos/: feature importance e matriz de confusão
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from config import CSV_PATH

OUTPUT_DIR = Path(__file__).parent / "graficos"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────── Carregamento e Feature Engineering ───────────────────────

def load_and_engineer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["screen_id"] = df["screen_id"].str.upper()
    df["dwell_sec"] = df["dwell_ms"] / 1000.0

    # Posição do evento dentro da sessão (ordem de navegação)
    df = df.sort_values(["session_id", "event_time_ms"])
    df["pos_na_sessao"] = df.groupby("session_id").cumcount()

    # Encoding da tela atual
    le_screen = LabelEncoder()
    df["screen_enc"] = le_screen.fit_transform(df["screen_id"])

    # Encoding do dia da semana
    dias = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["dia_enc"] = df["dia_semana"].map(dias).fillna(0).astype(int)

    # Encoding do perfil do usuário (se disponível)
    if "perfil_usuario" in df.columns:
        le_perfil = LabelEncoder()
        df["perfil_enc"] = le_perfil.fit_transform(df["perfil_usuario"].fillna("desconhecido"))
    else:
        df["perfil_enc"] = 0  # fallback: coluna ausente em dados legados

    # Próxima tela (dentro da mesma sessão)
    df["proxima_tela"] = (
        df.groupby("session_id")["screen_id"].shift(-1)
    )

    # Engajamento
    mediana = df["dwell_ms"].median()
    df["engajamento"] = (df["dwell_ms"] >= mediana).astype(int)  # 1=ALTO, 0=BAIXO

    return df, le_screen  # le_screen mantido para compatibilidade com dashboard


# ─────────────────────── Modelo 1: Classificador de Engajamento ───────────────────────

def treinar_engajamento(df: pd.DataFrame) -> dict:
    """
    Treina Random Forest para classificar ALTO (1) vs BAIXO (0) engajamento.
    Features: dwell_ms, hora_sessao, screen_enc, pos_na_sessao, dia_enc
    """
    features = ["dwell_ms", "hora_sessao", "screen_enc", "pos_na_sessao", "dia_enc", "perfil_enc"]
    X = df[features]
    y = df["engajamento"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    resultados = {
        "model":      model,
        "features":   features,
        "X_test":     X_test,
        "y_test":     y_test,
        "y_pred":     y_pred,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "cv_mean":    cv_scores.mean(),
        "cv_std":     cv_scores.std(),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report":     classification_report(y_test, y_pred, target_names=["BAIXO","ALTO"]),
        "nome":       "Engajamento (ALTO/BAIXO)",
    }
    return resultados


# ─────────────────────── Modelo 2: Preditor de Próxima Tela ───────────────────────

def treinar_proxima_tela(df: pd.DataFrame) -> dict:
    """
    Treina Random Forest para prever qual será a próxima tela.
    Remove linhas onde proxima_tela é NaN (último evento da sessão).
    """
    df_model = df.dropna(subset=["proxima_tela"]).copy()

    le_prox = LabelEncoder()
    df_model["proxima_enc"] = le_prox.fit_transform(df_model["proxima_tela"])

    features = ["screen_enc", "dwell_ms", "hora_sessao", "pos_na_sessao", "dia_enc", "perfil_enc"]
    X = df_model[features]
    y = df_model["proxima_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    resultados = {
        "model":      model,
        "le_prox":    le_prox,
        "features":   features,
        "X_test":     X_test,
        "y_test":     y_test,
        "y_pred":     y_pred,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":     recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":         f1_score(y_test, y_pred, average="macro", zero_division=0),
        "cv_mean":    cv_scores.mean(),
        "cv_std":     cv_scores.std(),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report":     classification_report(y_test, y_pred, target_names=le_prox.classes_),
        "nome":       "Próxima Tela (multiclass)",
        "classes":    le_prox.classes_,
    }
    return resultados


# ─────────────────────── Gráficos ───────────────────────

def plot_feature_importance(res: dict, sufixo: str) -> None:
    importances = res["model"].feature_importances_
    feat_df = pd.DataFrame({
        "feature":    res["features"],
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(feat_df["feature"], feat_df["importance"],
                   color="#4C9BE8", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_title(f"Feature Importance — {res['nome']}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importância")
    plt.tight_layout()
    path = OUTPUT_DIR / f"07_feature_importance_{sufixo}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[modelo_ml] Salvo: {path.name}")


def plot_confusion_matrix(res: dict, labels: list, sufixo: str) -> None:
    cm = res["conf_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Matriz de Confusão — {res['nome']}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    plt.tight_layout()
    path = OUTPUT_DIR / f"08_confusion_matrix_{sufixo}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[modelo_ml] Salvo: {path.name}")


# ─────────────────────── Print de métricas ───────────────────────

def print_metricas(res: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  MODELO: {res['nome']}")
    print(f"{'─'*55}")
    print(f"  Acurácia          : {res['accuracy']:.4f}  ({res['accuracy']*100:.1f}%)")
    print(f"  Precisão          : {res['precision']:.4f}")
    print(f"  Recall            : {res['recall']:.4f}")
    print(f"  F1-Score          : {res['f1']:.4f}")
    print(f"  CV (5-fold) média : {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
    print(f"\n  Relatório completo:\n")
    print(res["report"])


# ─────────────────────── Main ───────────────────────

def run() -> tuple:
    if not CSV_PATH.exists():
        print("[modelo_ml] CSV não encontrado. Execute gera_dados.py primeiro.")
        sys.exit(1)

    df, le_screen = load_and_engineer(CSV_PATH)

    print("\n════ TREINANDO MODELOS DE MACHINE LEARNING ════\n")

    # Modelo 1
    res_eng = treinar_engajamento(df)
    print_metricas(res_eng)
    plot_feature_importance(res_eng, "engajamento")
    plot_confusion_matrix(res_eng, ["BAIXO", "ALTO"], "engajamento")

    # Modelo 2
    res_prox = treinar_proxima_tela(df)
    print_metricas(res_prox)
    plot_feature_importance(res_prox, "proxima_tela")
    plot_confusion_matrix(res_prox, list(res_prox["classes"]), "proxima_tela")

    return res_eng, res_prox, df, le_screen


if __name__ == "__main__":
    run()
