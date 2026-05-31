"""
Phase 3 — Construction de la Topologie du Graphe
Tâche H : Matrice de corrélation de Pearson (sur train uniquement)
Tâche I : Seuil de corrélation -> Matrice d'adjacence
Tâche J : Visualisation du graphe avec NetworkX
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "graph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# TÂCHE H — Chargement des données train + Matrice de Pearson
# ---------------------------------------------------------------------------
print("=" * 50)
print("TÂCHE H — MATRICE DE CORRÉLATION DE PEARSON")
print("=" * 50)

print("\nChargement de load_hourly.csv.gz ...")
load_df = pd.read_csv(PROCESSED_DIR / "load_hourly.csv.gz", parse_dates=["datetime"])

client_cols = [c for c in load_df.columns if c not in ("datetime", "load_total_kw")]
print(f"  Clients : {len(client_cols)} | Heures totales : {len(load_df)}")

train_ratio = 0.70
train_end = int(len(load_df) * train_ratio)
train_df = load_df.iloc[:train_end][client_cols]
print(f"  Heures train utilisées : {train_end}")
print(
    f"  Période : {load_df['datetime'].iloc[0]} -> {load_df['datetime'].iloc[train_end-1]}"
)

# Supprimer les clients avec variance nulle
std_per_client = train_df.std()
clients_zero_var = std_per_client[std_per_client == 0].index.tolist()
if clients_zero_var:
    print(f"\n  ⚠️  Clients à variance nulle exclus : {len(clients_zero_var)}")
    train_df = train_df.drop(columns=clients_zero_var)
client_cols_valid = train_df.columns.tolist()
print(f"  Clients valides : {len(client_cols_valid)}")

# Calcul de la matrice de corrélation
print("\nCalcul de la matrice de corrélation ...")
values = train_df.values.astype(np.float64)
corr_array = np.corrcoef(values.T).astype(np.float32)
corr_array = np.nan_to_num(corr_array, nan=0.0)

corr_no_diag = corr_array.copy()
np.fill_diagonal(corr_no_diag, 0)
print(f"  Shape : {corr_array.shape}")
print(f"  Min hors diagonale : {corr_no_diag.min():.4f}")
print(f"  Max hors diagonale : {corr_no_diag.max():.4f}")
print(f"  Moyenne hors diagonale : {corr_no_diag.mean():.4f}")

np.save(OUTPUT_DIR / "correlation_matrix.npy", corr_array)
with open(OUTPUT_DIR / "client_cols_valid.txt", "w") as f:
    f.write("\n".join(client_cols_valid))
print("\nTâche H terminée avec succès !")

# ---------------------------------------------------------------------------
# TÂCHE I — Seuil de corrélation + Matrice d'adjacence
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TÂCHE I — SEUIL DE CORRÉLATION + MATRICE D'ADJACENCE")
print("=" * 50)

n_nodes = len(client_cols_valid)
THRESHOLD = 0.85
adj_matrix = (corr_array >= THRESHOLD).astype(np.uint8)
np.fill_diagonal(adj_matrix, 0)

n_edges = int(adj_matrix.sum()) // 2
density = n_edges / (n_nodes * (n_nodes - 1) / 2)
isolated = int((adj_matrix.sum(axis=1) == 0).sum())

print(f"\n  Seuil retenu : {THRESHOLD}")
print(f"  Nœuds : {n_nodes}")
print(f"  Arêtes : {n_edges}")
print(f"  Densité : {density:.4f} ({density*100:.2f}%)")
print(f"  Nœuds isolés : {isolated} / {n_nodes}")

np.save(OUTPUT_DIR / "adjacency_matrix.npy", adj_matrix)
np.savetxt(OUTPUT_DIR / "adjacency_matrix.csv", adj_matrix, delimiter=",", fmt="%d")

graph_info = pd.DataFrame(
    {
        "parametre": [
            "seuil_correlation",
            "n_noeuds",
            "n_aretes",
            "densite",
            "noeuds_isoles",
        ],
        "valeur": [THRESHOLD, n_nodes, n_edges, round(density, 4), isolated],
    }
)
graph_info.to_csv(OUTPUT_DIR / "graph_info.csv", index=False)
print("\nTâche I terminée avec succès !")

# ---------------------------------------------------------------------------
# TÂCHE J — Visualisation du graphe avec NetworkX
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TÂCHE J — VISUALISATION DU GRAPHE")
print("=" * 50)

# Construction du graphe NetworkX depuis la matrice d'adjacence
print("\nConstruction du graphe NetworkX ...")
G = nx.from_numpy_array(adj_matrix)

# Renommer les nœuds avec les vrais noms des clients
mapping = {i: client_cols_valid[i] for i in range(n_nodes)}
G = nx.relabel_nodes(G, mapping)

print(f"  Nœuds : {G.number_of_nodes()}")
print(f"  Arêtes : {G.number_of_edges()}")
print(f"  Nœuds isolés : {len(list(nx.isolates(G)))}")

# Statistiques du graphe
degrees = [d for _, d in G.degree()]
print(f"  Degré moyen : {np.mean(degrees):.2f}")
print(f"  Degré max   : {np.max(degrees)}")
print(f"  Degré min   : {np.min(degrees)}")

# Composantes connexes
components = list(nx.connected_components(G))
print(f"  Composantes connexes : {len(components)}")
print(f"  Plus grande composante : {max(len(c) for c in components)} nœuds")

# ---- Figure 1 : Visualisation globale du graphe ----
print("\nGénération des visualisations ...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Sous-graphe : composante principale uniquement (plus lisible)
largest_cc = max(components, key=len)
G_main = G.subgraph(largest_cc).copy()
pos_main = nx.spring_layout(G_main, seed=42, k=0.3)
deg_main = np.array([G_main.degree(n) for n in G_main.nodes()])

nx.draw_networkx_nodes(
    G_main,
    pos_main,
    ax=axes[0],
    node_size=20,
    node_color=deg_main,
    cmap=plt.cm.YlOrRd,
    alpha=0.85,
)
nx.draw_networkx_edges(G_main, pos_main, ax=axes[0], alpha=0.15, width=0.5)
axes[0].set_title(
    f"Composante principale ({len(largest_cc)} nœuds)\nSeuil corrélation = {THRESHOLD}",
    fontsize=13,
)
axes[0].axis("off")

# ---- Figure 2 : Distribution des degrés ----
axes[1].hist(degrees, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[1].axvline(
    np.mean(degrees),
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Moyenne : {np.mean(degrees):.1f}",
)
axes[1].set_title("Distribution des degrés des nœuds", fontsize=13)
axes[1].set_xlabel("Degré (nombre de voisins)")
axes[1].set_ylabel("Nombre de nœuds")
axes[1].legend()

plt.suptitle(
    f"Graphe de similarité — OPxGNN\n"
    f"{n_nodes} nœuds | {n_edges} arêtes | Seuil Pearson = {THRESHOLD}",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

# Sauvegarde
viz_path = OUTPUT_DIR / "graph_visualization.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
print(f"  -> Sauvegardé : {viz_path}")
plt.show()

print("\nTâche J terminée avec succès !")
print("\n" + "=" * 50)
print("PHASE 3 COMPLÈTE ✅")
print("=" * 50)
print(f"  Fichiers dans : {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
