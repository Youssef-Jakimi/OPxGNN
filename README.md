# OPxGNN — Optimisation de la distribution d’énergie (ML tabulaire vs GNN)

Projet étudiant : comparaison d’un baseline ML tabulaire (XGBoost) et d’un modèle GNN (GCN via PyTorch Geometric) pour détecter des états de **congestion** dans un réseau de distribution (370 clients).

## Idée générale (pipeline)
1. **Données brutes** : consommation électrique par client toutes les 15 minutes (2011–2014)
2. **Prétraitement** : passage en **horaire** (1h), gestion des anomalies (DST), gestion des clients apparaissant en cours d’historique
3. **Cible (classification)** : état `1` si la consommation dépasse un seuil de capacité (quantile), sinon `0`
4. **Découpage temporel** (anti-fuite) : train (2011–2013), val (2014 S1), test (2014 S2)
5. **Modèles** :
   - Baseline tabulaire : XGBoost (features à définir : lags, stats, etc.)
   - GNN : GCN sur un graphe de similarité (corrélations de profils)

## Pré-requis
- Python 3.10+ recommandé

## Installation
Dans PowerShell (Windows) :
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### Note importante (PyTorch Geometric)
`torch-geometric` peut nécessiter des wheels spécifiques (selon CPU/CUDA et version de Torch).
Si l’installation via `pip install -r requirements.txt` échoue, installez d’abord `torch`, puis suivez les instructions officielles PyG adaptées à votre configuration.

## Dataset (non versionné)
Le fichier brut `dataset/LD2011_2014.txt` fait ~711 MB et est **ignoré par Git** (voir `.gitignore`).
- Téléchargez le dataset *Electricity Load Diagrams 2011–2014* (UCI)
- Placez le fichier ici : `dataset/LD2011_2014.txt`

## Lancer le prétraitement
Script : `scripts/01_preprocess_ld2011_2014.py`

Commande minimale :
```bash
python scripts/01_preprocess_ld2011_2014.py
```

Sorties (créées dans `data/processed/`) :
- `load_hourly.csv.gz` : consommation horaire par client
- `labels_hourly.csv.gz` : labels 0/1 par client et par heure
- `thresholds.csv` : seuil(s) utilisé(s) pour la création des labels

### Options utiles
- Schéma de labels (défaut : un seuil par client) :
```bash
python scripts/01_preprocess_ld2011_2014.py --label-scheme per_node_quantile --quantile 0.95
```
- Seuil global basé sur le quantile des maxima par client :
```bash
python scripts/01_preprocess_ld2011_2014.py --label-scheme global_from_node_max --quantile 0.95
```

## Découpage temporel
Par défaut :
- **Train** : jusqu’à `2013-12-31 23:00:00` (inclus)
- **Validation** : `2014-01-01` à `2014-06-30 23:00:00` (inclus)
- **Test** : après `2014-06-30 23:00:00`

Vous pouvez ajuster avec `--train-end` et `--val-end`.

## Conseils Git/GitHub (travail en groupe)
- Le dataset brut est exclu : chacun le télécharge localement.
- Créez une branche par fonctionnalité (`preprocess`, `baseline-xgb`, `gnn`, etc.).
- Évitez de committer des sorties lourdes (`data/processed/`, `outputs/`) : elles sont ignorées.

---
Livrables et documentation du projet : **en français**.
