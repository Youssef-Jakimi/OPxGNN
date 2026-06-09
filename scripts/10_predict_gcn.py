#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 - Tache S
Generer et sauvegarder les predictions finales du modele GCN sur l'ensemble Test.

Sorties principales:
- data/processed/gcn_predictions_test.csv
- data/processed/gcn_probabilities_test.csv
- data/processed/gcn_test_metrics.json

Le script charge un checkpoint GCN sauvegarde par scripts/07_train_gcn.py ou
scripts/09_tune_gcn.py, reconstruit le dataset PyG test, calcule les logits,
les probabilites et les classes predites, puis archive les resultats.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataelectricity"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SCRIPT_PREPARE_PYG = SCRIPT_DIR / "05_prepare_pyg_data.py"
SCRIPT_GCN_MODEL = SCRIPT_DIR / "06_gcn_model.py"

DEFAULT_CHECKPOINT = OUTPUT_DIR / "gcn_best_model.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "gcn_predictions_test.csv"
PROBABILITIES_PATH = OUTPUT_DIR / "gcn_probabilities_test.csv"
METRICS_PATH = OUTPUT_DIR / "gcn_test_metrics.json"


def charger_module(path: Path, module_name: str):
    """Charge un module Python local dont le nom de fichier commence par un chiffre."""
    if not path.exists():
        raise FileNotFoundError(f"Script introuvable: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def construire_loader_test(batch_size: int, num_workers: int):
    """Reconstruit le DataLoader PyG de test avec le meme mapping que l'entrainement."""
    module_prepare = charger_module(SCRIPT_PREPARE_PYG, "prepare_pyg_data")

    clients_valides = module_prepare.charger_clients_valides(GRAPH_DIR / "client_cols_valid.txt")
    edge_index = module_prepare.construire_edge_index(GRAPH_DIR / "adjacency_matrix.npy")
    indices_valides = module_prepare.extraire_indices_valides(
        DATASET_DIR / "train_data.csv",
        clients_valides,
    )

    test_path = DATASET_DIR / "sequences_test.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"Fichier de sequences test introuvable: {test_path}")

    ds_test = module_prepare.DatasetGraphTemporel(
        test_path,
        edge_index,
        indices_valides,
    )

    loader_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return ds_test, loader_test


def charger_modele(checkpoint_path: Path, device: torch.device):
    """Charge le modele GCN depuis un checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint GCN introuvable: {checkpoint_path}")

    module_model = charger_module(SCRIPT_GCN_MODEL, "gcn_model")
    GCNClassifier = module_model.GCNClassifier

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError("Checkpoint invalide: champ model_config absent.")

    model = GCNClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, model_config


def predire_test(model, loader_test, device: torch.device) -> Dict[str, np.ndarray]:
    """Calcule les predictions, probabilites et logits sur l'ensemble test."""
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    prob_all: List[List[float]] = []
    logits_all: List[List[float]] = []

    with torch.no_grad():
        for batch in loader_test:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_all.extend(batch.y.detach().cpu().numpy().astype(int).tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().astype(int).tolist())
            prob_all.extend(probs.detach().cpu().numpy().tolist())
            logits_all.extend(logits.detach().cpu().numpy().tolist())

    return {
        "y_true": np.asarray(y_true_all, dtype=np.int64),
        "y_pred": np.asarray(y_pred_all, dtype=np.int64),
        "probs": np.asarray(prob_all, dtype=np.float64),
        "logits": np.asarray(logits_all, dtype=np.float64),
    }


def calculer_metriques(y_true: np.ndarray, y_pred: np.ndarray, prob_congestion: np.ndarray) -> Dict[str, Any]:
    """Calcule les metriques finales de classification binaire."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_normal": int((y_true == 0).sum()),
        "support_congestion": int((y_true == 1).sum()),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_congestion))
    else:
        metrics["roc_auc"] = None

    denom_sensitivity = tp + fn
    denom_specificity = tn + fp
    metrics["sensitivity"] = float(tp / denom_sensitivity) if denom_sensitivity else 0.0
    metrics["specificity"] = float(tn / denom_specificity) if denom_specificity else 0.0

    return metrics


def sauvegarder_predictions(resultats: Dict[str, np.ndarray], metrics: Dict[str, Any], checkpoint_info: Dict[str, Any]) -> None:
    """Sauvegarde les predictions brutes, probabilites et metriques."""
    y_true = resultats["y_true"]
    y_pred = resultats["y_pred"]
    probs = resultats["probs"]
    logits = resultats["logits"]

    df_predictions = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_normal": probs[:, 0],
            "prob_congestion": probs[:, 1],
            "logit_normal": logits[:, 0],
            "logit_congestion": logits[:, 1],
        }
    )
    df_predictions.to_csv(PREDICTIONS_PATH, index=False)

    df_probabilities = df_predictions[
        ["sample_index", "prob_normal", "prob_congestion"]
    ].copy()
    df_probabilities.to_csv(PROBABILITIES_PATH, index=False)

    payload_metrics = {
        "checkpoint": checkpoint_info,
        "metrics": metrics,
        "n_predictions": int(len(y_true)),
        "outputs": {
            "predictions_csv": str(PREDICTIONS_PATH),
            "probabilities_csv": str(PROBABILITIES_PATH),
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload_metrics, f, indent=2, ensure_ascii=False)


def afficher_avertissement_debug(checkpoint: Dict[str, Any]) -> None:
    """Signale un checkpoint issu d'un entrainement trop court."""
    epoch = checkpoint.get("epoch")
    val_f1 = checkpoint.get("val_f1")
    if epoch is not None and int(epoch) <= 2:
        print(
            "[ATTENTION] Le checkpoint charge semble venir d'un run tres court "
            f"(epoch={epoch}, val_f1={val_f1}). "
            "Les fichiers generes valident le pipeline, mais ne constituent pas "
            "forcement les predictions finales scientifiques."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tache S: predictions finales GCN sur test.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Chemin du checkpoint GCN.")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch pour l'inference.")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers DataLoader.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device PyTorch.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demande mais indisponible.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 78)
    print("PHASE 6 - TACHE S: PREDICTIONS FINALES GCN SUR TEST")
    print("=" * 78)
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Device: {device}")

    ds_test, loader_test = construire_loader_test(args.batch_size, args.num_workers)
    model, checkpoint, model_config = charger_modele(args.checkpoint, device)
    afficher_avertissement_debug(checkpoint)

    resultats = predire_test(model, loader_test, device)
    metrics = calculer_metriques(
        resultats["y_true"],
        resultats["y_pred"],
        resultats["probs"][:, 1],
    )

    checkpoint_info = {
        "path": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "val_f1": checkpoint.get("val_f1"),
        "model_config": model_config,
    }
    sauvegarder_predictions(resultats, metrics, checkpoint_info)

    print(f"[INFO] Taille test: {len(ds_test)}")
    print(
        "[METRIQUES] "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | "
        f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
    )
    print(
        "[CONFUSION] "
        f"TN={metrics['tn']} | FP={metrics['fp']} | "
        f"FN={metrics['fn']} | TP={metrics['tp']}"
    )
    print(f"[SAVE] Predictions:  {PREDICTIONS_PATH}")
    print(f"[SAVE] Probabilites: {PROBABILITIES_PATH}")
    print(f"[SAVE] Metriques:    {METRICS_PATH}")
    print("\nTache S terminee - Predictions GCN sauvegardees.")


if __name__ == "__main__":
    main()
