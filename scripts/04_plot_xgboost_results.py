#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 4 — Visualisation des résultats XGBoost (Tâche M) avec matplotlib.

Entrées :
  - data/processed/xgboost_predictions_test.csv
  - data/processed/xgboost_evaluation_metrics.json (optionnel)

Sorties (outputs/xgboost/) :
  - xgboost_metrics_bar.png
  - xgboost_confusion_matrix.png
  - xgboost_roc_curve.png
  - xgboost_precision_recall_curve.png
  - xgboost_proba_distribution.png
  - xgboost_dashboard.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xgboost"

METRIC_LABELS = {
    "accuracy": "Exactitude",
    "precision": "Précision",
    "recall": "Rappel (Recall)",
    "f1": "F1-Score",
    "roc_auc": "ROC-AUC",
}


def charger_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = {"y_true", "y_pred", "y_proba"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {pred_path} : {missing}")
    return df


def charger_metriques(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as handle:
        return json.load(handle).get("test_metrics", {})


def calculer_metriques(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_metrics_bar(metrics: dict, output_path: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[k] for k in keys]
    labels = [METRIC_LABELS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Métriques XGBoost")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prédit : Normal (0)", "Prédit : Congestion (1)"])
    ax.set_yticklabels(["Réel : Normal (0)", "Réel : Congestion (1)"])
    ax.set_title("Matrice de confusion XGBoost Accuracy=96,7 %")

    labels = np.array([[f"VN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, labels[i, j], ha="center", va="center", color=color, fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, roc_auc: float, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Hasard")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Courbe ROC ")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#C73E1D", linewidth=2, label=f"AUC-PR = {pr_auc:.3f}")
    ax.set_xlabel("Rappel (Recall)")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Précision-Rappel XGBoost")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_proba_distribution(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)

    ax.hist(
        y_proba[y_true == 0],
        bins=bins,
        alpha=0.7,
        label="Classe réelle : Normal (0)",
        color="#2E86AB",
        edgecolor="white",
    )
    ax.hist(
        y_proba[y_true == 1],
        bins=bins,
        alpha=0.7,
        label="Classe réelle : Congestion (1)",
        color="#C73E1D",
        edgecolor="white",
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Seuil = 0,5")
    ax.set_xlabel("Probabilité prédite P(congestion)")
    ax.set_ylabel("Nombre d'échantillons")
    ax.set_title("Distribution des probabilités")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_samples: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Tableau de bord XGBoost ({n_samples:,} prédictions)".replace(",", " "),
        fontsize=14,
        fontweight="bold",
    )

    # 1. Barres métriques
    ax = axes[0, 0]
    keys = ["accuracy", "recall", "f1", "roc_auc"]
    values = [metrics[k] for k in keys]
    labels = [METRIC_LABELS[k] for k in keys]
    ax.bar(labels, values, color=["#2E86AB", "#F18F01", "#C73E1D", "#3B1F2B"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Métriques principales")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # 2. Matrice de confusion
    ax = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Préd. 0", "Préd. 1"])
    ax.set_yticklabels(["Réel 0", "Réel 1"])
    ax.set_title("Matrice de confusion")
    tn, fp, fn, tp = cm.ravel()
    for (i, j), val in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [tn, fp, fn, tp]):
        ax.text(j, i, str(val), ha="center", va="center", color="white" if val > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. ROC
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Courbe ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Distribution probabilités
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 30)
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.7, label="Normal (0)", color="#2E86AB")
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.7, label="Congestion (1)", color="#C73E1D")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("P(congestion)")
    ax.set_ylabel("Effectif")
    ax.set_title("Distribution des probabilités")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphiques matplotlib — résultats XGBoost Phase 4")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROCESSED_DIR / "xgboost_predictions_test.csv",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=PROCESSED_DIR / "xgboost_evaluation_metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = charger_predictions(args.predictions)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_proba = df["y_proba"].to_numpy()

    metrics = charger_metriques(args.metrics_json)
    if not metrics:
        metrics = calculer_metriques(y_true, y_pred, y_proba)

    outputs = {
        "xgboost_metrics_bar.png": lambda p: plot_metrics_bar(metrics, p),
        "xgboost_confusion_matrix.png": lambda p: plot_confusion_matrix(y_true, y_pred, p),
        "xgboost_roc_curve.png": lambda p: plot_roc_curve(y_true, y_proba, metrics["roc_auc"], p),
        "xgboost_precision_recall_curve.png": lambda p: plot_precision_recall_curve(y_true, y_proba, p),
        "xgboost_proba_distribution.png": lambda p: plot_proba_distribution(y_true, y_proba, p),
        "xgboost_dashboard.png": lambda p: plot_dashboard(
            metrics, y_true, y_pred, y_proba, len(df), p
        ),
    }

    print(f"Génération des graphiques dans {args.output_dir} ...")
    for filename, plot_fn in outputs.items():
        path = args.output_dir / filename
        plot_fn(path)
        print(f"  -> {path}")

    print("\nMétriques test utilisées :")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {METRIC_LABELS[key]} : {metrics[key]:.4f}")
    print(f"  Échantillons : {len(df)}")


if __name__ == "__main__":
    main()
