#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5 — Tâche P
Boucle d'entraînement GCN avec fonction de perte et optimiseur Adam.

Ce script:
- charge les datasets PyG (train/val/test) produits à la tâche N
- charge l'architecture GCN de la tâche O
- entraîne le modèle avec CrossEntropyLoss + Adam
- suit les métriques train/val par époque
- sauvegarde le meilleur modèle (selon F1 validation)

Exemples:
  python scripts/07_train_gcn.py --epochs 20 --batch-size 32
  python scripts/07_train_gcn.py --epochs 2 --batch-size 8 --max-batches-train 20 --max-batches-val 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch_geometric.loader import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataelectricity"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_PREPARE_PYG = SCRIPT_DIR / "05_prepare_pyg_data.py"
SCRIPT_GCN_MODEL = SCRIPT_DIR / "06_gcn_model.py"


def charger_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Script introuvable: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def construire_dataloaders(batch_size: int, num_workers: int):
    module_prepare = charger_module(SCRIPT_PREPARE_PYG, "prepare_pyg_data")

    clients_valides = module_prepare.charger_clients_valides(GRAPH_DIR / "client_cols_valid.txt")
    edge_index = module_prepare.construire_edge_index(GRAPH_DIR / "adjacency_matrix.npy")
    indices_valides = module_prepare.extraire_indices_valides(
        DATASET_DIR / "train_data.csv",
        clients_valides,
    )

    ds_train = module_prepare.DatasetGraphTemporel(
        DATASET_DIR / "sequences_train.npz",
        edge_index,
        indices_valides,
    )

    ds_val = None
    val_path = DATASET_DIR / "sequences_val.npz"
    if val_path.exists():
        ds_val = module_prepare.DatasetGraphTemporel(
            val_path,
            edge_index,
            indices_valides,
        )

    ds_test = None
    test_path = DATASET_DIR / "sequences_test.npz"
    if test_path.exists():
        ds_test = module_prepare.DatasetGraphTemporel(
            test_path,
            edge_index,
            indices_valides,
        )

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers) if ds_val else None
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers) if ds_test else None

    return ds_train, ds_val, ds_test, loader_train, loader_val, loader_test


def calculer_poids_classes(ds_train) -> torch.Tensor:
    y_train = ds_train.y.astype(np.int64)
    n_classes = 2
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        return torch.ones(n_classes, dtype=torch.float32)

    total = counts.sum()
    weights = total / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def afficher_distribution_labels(nom: str, dataset) -> None:
    if dataset is None:
        print(f"[DEBUG][{nom}] dataset absent")
        return

    labels = np.asarray(dataset.y, dtype=np.int64)
    valeurs, comptes = np.unique(labels, return_counts=True)
    distribution = {int(v): int(c) for v, c in zip(valeurs, comptes)}
    print(
        f"[DEBUG][{nom}] y_unique_counts={distribution} | "
        f"n={len(labels)} | n_pos={int((labels == 1).sum())} | "
        f"n_neg={int((labels == 0).sum())}"
    )
    if not np.all(np.isin(labels, [0, 1])):
        print(f"[WARN][{nom}] labels hors encodage binaire 0/1 detectes.")
    if int((labels == 1).sum()) == 0:
        print(f"[WARN][{nom}] aucune classe positive: precision/recall/F1 positifs seront non informatifs.")


def calculer_metriques_classification(y_true_all, y_pred_all) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
    }


def afficher_debug_predictions(
    nom: str,
    y_true_all,
    y_pred_all,
    prob_pos_samples,
    max_samples: int = 10,
) -> None:
    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)
    prob_pos = np.asarray(prob_pos_samples, dtype=np.float64)
    valeurs_y, comptes_y = np.unique(y_true, return_counts=True)
    valeurs_pred, comptes_pred = np.unique(y_pred, return_counts=True)

    print(
        f"[DEBUG][{nom}] y_true_counts={dict(zip(valeurs_y.tolist(), comptes_y.tolist()))} | "
        f"y_pred_counts={dict(zip(valeurs_pred.tolist(), comptes_pred.tolist()))}"
    )
    print(
        f"[DEBUG][{nom}] n_pos_labels={int((y_true == 1).sum())} | "
        f"n_pos_predictions={int((y_pred == 1).sum())}"
    )
    print(f"[DEBUG][{nom}] sample_prob_class1={np.round(prob_pos[:max_samples], 6).tolist()}")
    print(f"[DEBUG][{nom}] sample_pred_labels={y_pred[:max_samples].tolist()}")
    print(f"[DEBUG][{nom}] sample_true_labels={y_true[:max_samples].tolist()}")


def verifier_validation(ds_val) -> None:
    if ds_val is None:
        return

    y_val = np.asarray(ds_val.y, dtype=np.int64)
    if int((y_val == 1).sum()) == 0:
        print(
            "[WARN] Validation contient uniquement la classe 0. "
            "Le F1 de la classe positive sera toujours 0 avec zero_division=0, "
            "meme si la perte validation est tres faible."
        )


def evaluer_modele(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    split_name: str = "eval",
    debug_metrics: bool = False,
) -> Dict[str, float]:
    model.eval()

    losses = []
    y_true_all = []
    y_pred_all = []
    prob_pos_samples = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            losses.append(float(loss.item()))
            y_true_all.extend(batch.y.detach().cpu().numpy().tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().tolist())
            if len(prob_pos_samples) < 10:
                prob_pos_samples.extend(probs[:, 1].detach().cpu().numpy().tolist())

    if not losses:
        return {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    if debug_metrics:
        afficher_debug_predictions(split_name, y_true_all, y_pred_all, prob_pos_samples)

    metrics = {"loss": float(np.mean(losses))}
    metrics.update(calculer_metriques_classification(y_true_all, y_pred_all))
    return metrics


def entrainer_une_epoque(
    model: nn.Module,
    loader_train,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    debug_metrics: bool = False,
) -> Dict[str, float]:
    model.train()

    losses = []
    y_true_all = []
    y_pred_all = []
    prob_pos_samples = []

    for i, batch in enumerate(loader_train):
        if max_batches is not None and i >= max_batches:
            break

        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)
        losses.append(float(loss.item()))
        y_true_all.extend(batch.y.detach().cpu().numpy().tolist())
        y_pred_all.extend(preds.detach().cpu().numpy().tolist())
        if len(prob_pos_samples) < 10:
            prob_pos_samples.extend(probs[:, 1].detach().cpu().numpy().tolist())

    if not losses:
        return {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    if debug_metrics:
        afficher_debug_predictions("train", y_true_all, y_pred_all, prob_pos_samples)

    metrics = {"loss": float(np.mean(losses))}
    metrics.update(calculer_metriques_classification(y_true_all, y_pred_all))
    return metrics


def main() -> None:
    # ---------------------------------------------------------
    # FLAG DE DÉBOGAGE / ENTRAÎNEMENT COMPLET
    # Mets FULL_TRAINING = True pour lancer le vrai entraînement
    # Mets FULL_TRAINING = False pour un run interactif rapide
    # ---------------------------------------------------------
    FULL_TRAINING = True

    parser = argparse.ArgumentParser(description="Tâche P: boucle d'entraînement GCN.")
    parser.add_argument("--epochs", type=int, default= 10 if FULL_TRAINING else 2, help="Nombre d'époques.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers DataLoader.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Dimension cachée GCN.")
    parser.add_argument("--num-gcn-layers", type=int, default=3, help="Nombre de couches GCN.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device d'entraînement.")
    
    # Remplacé dynamiquement par le flag FULL_TRAINING
    parser.add_argument("--max-batches-train", type=int, default=None if FULL_TRAINING else 20, help="Limite debug de batches train.")
    parser.add_argument("--max-batches-val", type=int, default=None if FULL_TRAINING else 5, help="Limite debug de batches val.")
    parser.add_argument("--debug-metrics", action=argparse.BooleanOptionalAction, default=True, help="Affiche des diagnostics temporaires labels/predictions.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    print("=" * 78)
    print("PHASE 5 — TÂCHE P: BOUCLE D'ENTRAÎNEMENT GCN (LOSS + ADAM)")
    print("=" * 78)

    module_model = charger_module(SCRIPT_GCN_MODEL, "gcn_model")
    GCNClassifier = module_model.GCNClassifier

    ds_train, ds_val, ds_test, loader_train, loader_val, loader_test = construire_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sample_batch = next(iter(loader_train))
    in_channels = int(sample_batch.x.shape[1])

    model = GCNClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    class_weights = calculer_poids_classes(ds_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[INFO] in_channels={in_channels}")
    print(f"[INFO] train_size={len(ds_train)} | val_size={len(ds_val) if ds_val else 0} | test_size={len(ds_test) if ds_test else 0}")
    print(f"[INFO] class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")
    if args.debug_metrics:
        afficher_distribution_labels("train", ds_train)
        afficher_distribution_labels("val", ds_val)
        afficher_distribution_labels("test", ds_test)
    verifier_validation(ds_val)

    historique = {
        "config": vars(args),
        "class_weights": class_weights.detach().cpu().numpy().tolist(),
        "epochs": [],
    }

    meilleur_f1_val = -1.0
    best_model_path = OUTPUT_DIR / "gcn_best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = entrainer_une_epoque(
            model,
            loader_train,
            optimizer,
            criterion,
            device,
            max_batches=args.max_batches_train,
            debug_metrics=args.debug_metrics and epoch == 1,
        )

        if loader_val is not None:
            val_metrics = evaluer_modele(
                model,
                loader_val,
                criterion,
                device,
                max_batches=args.max_batches_val,
                split_name="val",
                debug_metrics=args.debug_metrics and epoch == 1,
            )
        else:
            val_metrics = {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

        if loader_val is not None and val_metrics["f1"] > meilleur_f1_val:
            meilleur_f1_val = val_metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "in_channels": in_channels,
                    "hidden_channels": args.hidden_channels,
                    "num_gcn_layers": args.num_gcn_layers,
                    "dropout": args.dropout,
                    "num_classes": 2,
                },
                "epoch": epoch,
                "val_f1": float(meilleur_f1_val),
            }, best_model_path)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        historique["epochs"].append(row)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

    # Évaluation finale test avec le meilleur modèle val si disponible
    if loader_test is not None:
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluer_modele(
            model,
            loader_test,
            criterion,
            device,
            split_name="test",
            debug_metrics=args.debug_metrics,
        )
        historique["test"] = test_metrics
        print(
            "[TEST] "
            f"loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"prec={test_metrics['precision']:.4f} "
            f"rec={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )

    history_path = OUTPUT_DIR / "gcn_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(historique, f, indent=2)

    print(f"[SAVE] Historique: {history_path}")
    if best_model_path.exists():
        print(f"[SAVE] Meilleur modèle: {best_model_path}")

    print("\nTache P prete - Boucle d'entrainement implementee (Loss + Adam + metriques).")


if __name__ == "__main__":
    main()
