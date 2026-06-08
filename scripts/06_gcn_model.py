#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5 — Tâche O
Coder l'architecture GCN (PyTorch Geometric) pour la classification binaire
au niveau du graphe.

Ce script définit:
- une classe GCNClassifier (convolutions géométriques + activations + sortie)
- un test de passage avant (forward) sur un batch réel des données PyG

Exécution (test architecture seulement):
  python scripts/06_gcn_model.py --batch-size 8
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataelectricity"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
SCRIPT_PREPARE_PYG = SCRIPT_DIR / "05_prepare_pyg_data.py"


class GCNClassifier(nn.Module):
    """
    Classifieur GCN pour classification binaire de graphes.

    Entrée:
      - x: (nb_noeuds_total_batch, in_channels)
      - edge_index: (2, nb_aretes_total_batch)
      - batch: (nb_noeuds_total_batch,) index graphe par nœud

    Sortie:
      - logits: (nb_graphes_batch, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gcn_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if num_gcn_layers < 2:
            raise ValueError("num_gcn_layers doit être >= 2.")

        self.dropout = dropout

        # Bloc de convolutions GCN
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Tête de classification après pooling global
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Si batch absent (graphe unique), créer un batch artificiel de zéros
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Empilement GCN + ReLU + Dropout
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Agrégation graphe (readout) -> vecteur par graphe
        g = global_mean_pool(x, batch)

        # Logits finaux pour classification binaire
        logits = self.classifier(g)
        return logits


# -----------------------------------------------------------------------------
# Utilitaire pour charger les fonctions de préparation PyG depuis la tâche N
# -----------------------------------------------------------------------------
def charger_module_preparation():
    if not SCRIPT_PREPARE_PYG.exists():
        raise FileNotFoundError(f"Script introuvable: {SCRIPT_PREPARE_PYG}")

    spec = importlib.util.spec_from_file_location("prepare_pyg_data", SCRIPT_PREPARE_PYG)
    if spec is None or spec.loader is None:
        raise ImportError("Impossible de charger le module de préparation PyG.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def construire_loader_train(batch_size: int, num_workers: int):
    module = charger_module_preparation()

    clients_valides = module.charger_clients_valides(GRAPH_DIR / "client_cols_valid.txt")
    edge_index = module.construire_edge_index(GRAPH_DIR / "adjacency_matrix.npy")
    indices_valides = module.extraire_indices_valides(
        DATASET_DIR / "train_data.csv",
        clients_valides,
    )

    ds_train = module.DatasetGraphTemporel(
        DATASET_DIR / "sequences_train.npz",
        edge_index,
        indices_valides,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return loader_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Tâche O: architecture GCN PyTorch Geometric.")
    parser.add_argument("--batch-size", type=int, default=8, help="Taille de batch pour test forward.")
    parser.add_argument("--num-workers", type=int, default=0, help="Nombre de workers DataLoader.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Dimension cachée GCN.")
    parser.add_argument("--num-gcn-layers", type=int, default=3, help="Nombre de couches GCN.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout entre couches.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device PyTorch.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible sur cette machine.")

    print("=" * 70)
    print("PHASE 5 — TÂCHE O: ARCHITECTURE GCN")
    print("=" * 70)

    device = torch.device(args.device)
    loader_train = construire_loader_train(args.batch_size, args.num_workers)

    batch = next(iter(loader_train)).to(device)
    in_channels = int(batch.x.shape[1])

    model = GCNClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    print(f"[OK] in_channels: {in_channels}")
    print(f"[OK] batch.x: {tuple(batch.x.shape)}")
    print(f"[OK] batch.edge_index: {tuple(batch.edge_index.shape)}")
    print(f"[OK] batch.y: {tuple(batch.y.shape)}")
    print(f"[OK] logits: {tuple(logits.shape)}")
    print(f"[OK] probs:  {tuple(probs.shape)}")

    if logits.shape[0] != batch.y.shape[0] or logits.shape[1] != 2:
        raise RuntimeError("Sortie modèle invalide: dimensions inattendues des logits.")

    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Paramètres entraînables: {nb_params}")
    print("\nTâche O prête ✅ Architecture GCN implémentée et validée en forward.")


if __name__ == "__main__":
    main()
