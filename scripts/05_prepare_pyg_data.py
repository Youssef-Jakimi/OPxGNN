#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5 — Tâche N
Transformer les séries temporelles et la matrice d'adjacence en objets Data
compatibles avec PyTorch Geometric.

Entrées attendues:
- dataset/dataelectricity/sequences_train.npz
- dataset/dataelectricity/sequences_val.npz (optionnel à ce stade)
- dataset/dataelectricity/sequences_test.npz
- dataset/dataelectricity/train_data.csv
- data/graph/adjacency_matrix.npy
- data/graph/client_cols_valid.txt

Sortie:
- Objets Data (PyG) construits à la volée via Dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataelectricity"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"


def charger_clients_valides(client_cols_valid_path: Path) -> List[str]:
    with open(client_cols_valid_path, "r", encoding="utf-8") as f:
        clients = [line.strip() for line in f if line.strip()]
    if not clients:
        raise ValueError("La liste des clients valides est vide.")
    return clients


def construire_edge_index(adjacency_path: Path) -> torch.Tensor:
    adj = np.load(adjacency_path)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Matrice d'adjacence invalide: shape={adj.shape}")

    src, dst = np.where(adj > 0)
    if len(src) == 0:
        raise ValueError("Aucune arête trouvée dans la matrice d'adjacence.")

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return edge_index


def extraire_indices_valides(train_csv_path: Path, clients_valides: List[str]) -> np.ndarray:
    df_train = pd.read_csv(train_csv_path, nrows=1)
    colonnes_clients = [c for c in df_train.columns if c.startswith("MT_")]

    if not colonnes_clients:
        raise ValueError("Aucune colonne MT_* trouvée dans train_data.csv.")

    index_map = {c: i for i, c in enumerate(colonnes_clients)}
    indices = [index_map[c] for c in clients_valides if c in index_map]

    if len(indices) != len(clients_valides):
        manquants = [c for c in clients_valides if c not in index_map]
        preview = ", ".join(manquants[:5])
        raise ValueError(
            "Mapping incomplet des clients valides vers les colonnes de séquences. "
            f"Manquants={len(manquants)}; exemples: {preview}"
        )

    return np.array(indices, dtype=np.int64)


class DatasetGraphTemporel(Dataset):
    """
    Chaque échantillon temporel devient un graphe PyG:
    - x: (n_noeuds, n_features_noeud) = (348, 24)
    - edge_index: (2, n_aretes_orientees)
    - y: label graphe binaire (0/1)
    """

    def __init__(self, npz_path: Path, edge_index: torch.Tensor, indices_valides: np.ndarray) -> None:
        self.npz_path = npz_path
        data = np.load(npz_path)

        if "X" not in data or "y_cls" not in data:
            raise ValueError(f"Fichier NPZ invalide ({npz_path.name}): clés attendues X et y_cls.")

        self.X = data["X"]  # (N, T, C)
        self.y = data["y_cls"]  # (N,)
        self.edge_index = edge_index
        self.indices_valides = indices_valides

        if self.X.ndim != 3:
            raise ValueError(f"X doit être 3D (N,T,C). Reçu: {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y_cls doit être 1D (N,). Reçu: {self.y.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Incohérence entre nombre d'échantillons X et y_cls.")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Data:
        # x brut en (T, C=370)
        x_raw = self.X[idx]

        # Filtrage vers les clients valides du graphe: (T, 348)
        x_valide = x_raw[:, self.indices_valides]

        # PyG attend (n_noeuds, n_features_noeud): (348, 24)
        x = torch.tensor(x_valide.T, dtype=torch.float32)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)

        return Data(x=x, edge_index=self.edge_index, y=y)


def construire_dataset_si_existe(npz_path: Path, edge_index: torch.Tensor, indices_valides: np.ndarray):
    if not npz_path.exists():
        print(f"[INFO] Fichier absent, dataset ignoré: {npz_path}")
        return None
    return DatasetGraphTemporel(npz_path, edge_index, indices_valides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare les objets Data PyG pour la tâche N.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch pour le test DataLoader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Nombre de workers DataLoader.")
    parser.add_argument("--sanity-samples", type=int, default=2, help="Nombre d'échantillons à vérifier.")
    args = parser.parse_args()

    adjacency_path = GRAPH_DIR / "adjacency_matrix.npy"
    clients_valides_path = GRAPH_DIR / "client_cols_valid.txt"
    train_csv_path = DATASET_DIR / "train_data.csv"

    seq_train_path = DATASET_DIR / "sequences_train.npz"
    seq_val_path = DATASET_DIR / "sequences_val.npz"
    seq_test_path = DATASET_DIR / "sequences_test.npz"

    print("=" * 70)
    print("PHASE 5 — TÂCHE N: PRÉPARATION DES OBJETS Data (PyTorch Geometric)")
    print("=" * 70)

    for p in [adjacency_path, clients_valides_path, train_csv_path, seq_train_path, seq_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier requis introuvable: {p}")

    clients_valides = charger_clients_valides(clients_valides_path)
    edge_index = construire_edge_index(adjacency_path)
    indices_valides = extraire_indices_valides(train_csv_path, clients_valides)

    print(f"[OK] Clients valides: {len(clients_valides)}")
    print(f"[OK] edge_index shape: {tuple(edge_index.shape)}")

    ds_train = construire_dataset_si_existe(seq_train_path, edge_index, indices_valides)
    ds_val = construire_dataset_si_existe(seq_val_path, edge_index, indices_valides)
    ds_test = construire_dataset_si_existe(seq_test_path, edge_index, indices_valides)

    if ds_train is None:
        raise FileNotFoundError("sequences_train.npz est requis pour continuer.")

    print(f"[OK] Taille dataset train: {len(ds_train)}")
    if ds_val is not None:
        print(f"[OK] Taille dataset val: {len(ds_val)}")
    if ds_test is not None:
        print(f"[OK] Taille dataset test: {len(ds_test)}")

    # Sanity checks sur quelques échantillons
    nb_check = min(args.sanity_samples, len(ds_train))
    for i in range(nb_check):
        sample = ds_train[i]
        print(
            f"[CHECK {i}] x={tuple(sample.x.shape)} | "
            f"edge_index={tuple(sample.edge_index.shape)} | y={int(sample.y.item())}"
        )

    # Test DataLoader pour la suite (tâches O/P)
    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    premier_batch = next(iter(loader_train))
    print(
        "[BATCH] "
        f"x={tuple(premier_batch.x.shape)} | "
        f"edge_index={tuple(premier_batch.edge_index.shape)} | "
        f"y={tuple(premier_batch.y.shape)} | "
        f"batch={tuple(premier_batch.batch.shape)}"
    )

    print("\nTâche N prête ✅ Les objets Data PyG sont correctement construits.")


if __name__ == "__main__":
    main()
