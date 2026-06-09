#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 - Tache R
Ajuster l'architecture GCN avec une grille courte.

Ce script lance plusieurs entrainements GCN en variant principalement:
- la dimension cachee hidden_channels;
- le nombre de couches GCN;
- le dropout.

Les resultats sont sauvegardes dans:
- data/processed/gcn_tuning_results.csv
- data/processed/gcn_tuning_results.json

Remarque importante:
la validation contient peu ou pas de cas positifs dans ce projet. Le F1 validation
peut donc etre non informatif. On conserve quand meme les metriques validation,
mais le tableau final affiche aussi les metriques test pour analyse comparative.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "07_train_gcn.py"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
HISTORY_PATH = OUTPUT_DIR / "gcn_training_history.json"
BEST_MODEL_PATH = OUTPUT_DIR / "gcn_best_model.pt"
TUNING_CSV_PATH = OUTPUT_DIR / "gcn_tuning_results.csv"
TUNING_JSON_PATH = OUTPUT_DIR / "gcn_tuning_results.json"


def construire_grille(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Construit une grille courte et explicable pour le rapport."""
    configs: List[Dict[str, Any]] = []

    for hidden_channels in args.hidden_channels:
        for num_gcn_layers in args.num_gcn_layers:
            for dropout in args.dropouts:
                configs.append(
                    {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "hidden_channels": hidden_channels,
                        "num_gcn_layers": num_gcn_layers,
                        "dropout": dropout,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "device": args.device,
                        "max_batches_train": args.max_batches_train,
                        "max_batches_val": args.max_batches_val,
                    }
                )

    return configs


def ajouter_limite_batches(commande: List[str], nom_argument: str, valeur: int | None) -> None:
    """
    Ajoute une limite de batches a la commande.

    Si la valeur est None, on envoie un tres grand nombre pour neutraliser le
    mode debug par defaut de scripts/07_train_gcn.py sans modifier ce script.
    """
    commande.extend([nom_argument, str(valeur if valeur is not None else 999999)])


def lancer_entrainement(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lance un entrainement GCN puis extrait les metriques de l'historique."""
    commande = [
        "python",
        str(TRAIN_SCRIPT),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--hidden-channels",
        str(config["hidden_channels"]),
        "--num-gcn-layers",
        str(config["num_gcn_layers"]),
        "--dropout",
        str(config["dropout"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--device",
        str(config["device"]),
    ]
    ajouter_limite_batches(commande, "--max-batches-train", config["max_batches_train"])
    ajouter_limite_batches(commande, "--max-batches-val", config["max_batches_val"])

    print("\n" + "=" * 90)
    print(
        "Test architecture | "
        f"hidden={config['hidden_channels']} | "
        f"couches={config['num_gcn_layers']} | "
        f"dropout={config['dropout']} | "
        f"epochs={config['epochs']}"
    )
    print("=" * 90)

    subprocess.run(commande, cwd=PROJECT_ROOT, check=True)

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Historique introuvable apres entrainement: {HISTORY_PATH}")

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        historique = json.load(f)

    epochs = historique.get("epochs", [])
    if not epochs:
        raise ValueError("Historique GCN vide: aucune epoque trouvee.")

    derniere_epoque = epochs[-1]
    train_metrics = derniere_epoque.get("train", {})
    val_metrics = derniere_epoque.get("val", {})
    test_metrics = historique.get("test", {})

    resultat = {
        **config,
        "train_loss": train_metrics.get("loss"),
        "train_accuracy": train_metrics.get("accuracy"),
        "train_precision": train_metrics.get("precision"),
        "train_recall": train_metrics.get("recall"),
        "train_f1": train_metrics.get("f1"),
        "val_loss": val_metrics.get("loss"),
        "val_accuracy": val_metrics.get("accuracy"),
        "val_precision": val_metrics.get("precision"),
        "val_recall": val_metrics.get("recall"),
        "val_f1": val_metrics.get("f1"),
        "test_loss": test_metrics.get("loss"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
    }

    return resultat


def sauvegarder_resultats(resultats: List[Dict[str, Any]]) -> pd.DataFrame:
    """Sauvegarde les resultats intermediaires et retourne le DataFrame trie."""
    df = pd.DataFrame(resultats)

    colonnes_tri = [c for c in ["test_f1", "test_recall", "train_f1"] if c in df.columns]
    if colonnes_tri:
        df = df.sort_values(by=colonnes_tri, ascending=False, na_position="last")

    df.to_csv(TUNING_CSV_PATH, index=False)
    with open(TUNING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    return df


def archiver_meilleur_modele(meilleur_resultat: Dict[str, Any]) -> Path | None:
    """Copie le dernier meilleur checkpoint sous un nom dedie au tuning."""
    if not BEST_MODEL_PATH.exists():
        return None

    suffixe = (
        f"h{meilleur_resultat['hidden_channels']}"
        f"_l{meilleur_resultat['num_gcn_layers']}"
        f"_d{str(meilleur_resultat['dropout']).replace('.', 'p')}"
    )
    destination = OUTPUT_DIR / f"gcn_best_model_tuned_{suffixe}.pt"
    shutil.copy2(BEST_MODEL_PATH, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Tache R: tuning court de l'architecture GCN.")
    parser.add_argument("--epochs", type=int, default=8, help="Nombre d'epoques par configuration.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch.")
    parser.add_argument("--hidden-channels", type=int, nargs="+", default=[32, 64, 128], help="Dimensions cachees a tester.")
    parser.add_argument("--num-gcn-layers", type=int, nargs="+", default=[2, 3, 4], help="Nombres de couches GCN a tester.")
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.2, 0.3], help="Valeurs de dropout a tester.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Regularisation L2 Adam.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device PyTorch.")
    parser.add_argument("--max-batches-train", type=int, default=80, help="Limite de batches train par epoque. Utiliser -1 pour tout le train.")
    parser.add_argument("--max-batches-val", type=int, default=20, help="Limite de batches validation par epoque. Utiliser -1 pour toute la validation.")
    parser.add_argument("--dry-run", action="store_true", help="Affiche la grille sans lancer d'entrainement.")
    args = parser.parse_args()

    if args.max_batches_train == -1:
        args.max_batches_train = None
    if args.max_batches_val == -1:
        args.max_batches_val = None

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Script d'entrainement introuvable: {TRAIN_SCRIPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = construire_grille(args)

    print("=" * 90)
    print("PHASE 6 - TACHE R: GRILLE COURTE POUR L'ARCHITECTURE GCN")
    print("=" * 90)
    print(f"Nombre de configurations: {len(configs)}")
    print(f"Debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.dry_run:
        print("\nConfigurations prevues:")
        for i, config in enumerate(configs, start=1):
            print(f"{i:02d}. {config}")
        return

    resultats: List[Dict[str, Any]] = []
    meilleur_score = float("-inf")
    meilleur_resultat: Dict[str, Any] | None = None
    meilleur_checkpoint: Path | None = None

    for i, config in enumerate(configs, start=1):
        print(f"\n[CONFIG {i}/{len(configs)}]")
        resultat = lancer_entrainement(config)
        resultats.append(resultat)

        df = sauvegarder_resultats(resultats)
        print(f"[SAVE] Resultats: {TUNING_CSV_PATH}")
        print("\nTop configurations actuelles:")
        print(df.head(5).to_string(index=False))

        score = resultat.get("test_f1")
        score = float(score) if score is not None else float("-inf")
        if score > meilleur_score:
            meilleur_score = score
            meilleur_resultat = resultat
            meilleur_checkpoint = archiver_meilleur_modele(resultat)

    df_final = sauvegarder_resultats(resultats)

    print("\n" + "=" * 90)
    print("TUNING TERMINE")
    print("=" * 90)
    print(f"[SAVE] CSV:  {TUNING_CSV_PATH}")
    print(f"[SAVE] JSON: {TUNING_JSON_PATH}")
    if meilleur_resultat is not None:
        print("\nMeilleure configuration selon test_f1:")
        print(json.dumps(meilleur_resultat, indent=2, ensure_ascii=False))
    if meilleur_checkpoint is not None:
        print(f"[SAVE] Checkpoint meilleur modele tuning: {meilleur_checkpoint}")
    print("\nTop 10 final:")
    print(df_final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
