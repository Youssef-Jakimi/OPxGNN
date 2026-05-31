# État d'avancement de l'équipe

Date: 2026-05-29

## Réalisations confirmées
- Dépôt Git initialisé, .gitignore en place et requirements.txt configuré.
- Dataset brut présent localement dans dataset/LD2011_2014.txt.
- Script de prétraitement réalisé : import du fichier, rééchantillonnage horaire, création des labels et split chronologique.

### Rapport de session - [29 mai 2026] / [IBTIHAAAL]

* **Tâche(s) traitée(s) :**
  Phase 2 complète — **Tâche D** (import pandas `;` / `,`), **Tâche E** (rééchantillonnage 15 min → 1 h + fillna), **Tâche F** (target classification charge totale + seuil 95e centile), **Tâche G** (split chronologique strict). Exécution, débogage, optimisation mémoire/disque et finalisation de `scripts/01_preprocess_ld2011_2014.py`.

* **Ce qui a été accompli :**
  - **Chargement (D)** : `read_csv(sep=';', decimal=',', parse_dates=[0])` sur `LD2011_2014.txt` — 140 256 × 371 (datetime + 370 clients MT_xxx).
  - **Rééchantillonnage (E)** : `set_index('datetime').resample('1h').mean().fillna(0)` — **35 065 lignes horaires** (÷4 vs 15 min). 0 NaN avant/après resample.
  - **Labels classification (F)** : `load_total_kw = sum(370 clients)` ; seuil = **`np.percentile(load_total_train, 95) = 360 203,93 kW`** (calculé **uniquement sur train**) ; `label = 1` si `load_total > seuil`, sinon `0`. Résultat : **1 827 / 35 065 congestés (5,21 %)** ; train : 1 228 / 24 545.
  - **Split chronologique (G)** : 70 / 15 / 15 par `iloc` sans shuffle — Train 24 545 (→ 2013-10-19 16h), Val 5 259, Test 5 261.
  - **Normalisation** : `MinMaxScaler` fitté **sur train uniquement** ; transform sur tout le dataset (val/test peuvent dépasser [0,1]).
  - **Séquences temporelles** : fenêtre **`n_steps=24`** (24 h à pas horaire) ; shape **`X: (N, 24, 370)`**, **`y_cls: (N,)`** (classification), **`y: (N, 370)`** (régression next-step conservée). Totaux : 35 041 séquences — train 24 528, val 5 256, test 5 257.
  - **CLI script** : `--sequences-only`, `--skip-existing`, `--export-x-csv` (désactivé par défaut).
  - **Artefacts vérifiés** : shapes `.npz` confirmées après refonte horaire.

* **Fichiers créés ou modifiés :**
  - **Modifié** : `OPxGNN/scripts/01_preprocess_ld2011_2014.py`
  - **`OPxGNN/data/processed/`** (artefacts projet, à partager via Drive) :
    - `load_hourly.csv.gz` (~79 Mo) — datetime, 370 clients kW bruts, `load_total_kw`
    - `labels_hourly.csv.gz` (~0,4 Mo) — datetime, `load_total_kw`, `label`
    - `thresholds.csv` — percentile=95, threshold_kw=360203.93, n_train=24545
  - **`OPxGNN/dataset/dataelectricity/`** (exécution locale, à partager via Drive) :
    - `train_data.csv`, `validation_data.csv`, `test_data.csv`, `electricity_normalized_full.csv`, `scaler_minmax.pkl`
    - `sequences_{train,val,test}.npz` — clés `X`, `y`, `y_cls` (~556 / 154 / 157 Mo)
    - `y_{train,val,test}.csv`, `y_cls_{train,val,test}.csv`
  - **Non générés** : `X_*.csv` aplatis (~20+ Go chacun, omis volontairement)

* **Problèmes rencontrés / Choix techniques :**
  1. **Virgules décimales** → colonnes en `str` ; fix `decimal=','`.
  2. **RAM insuffisante** (1re génération séquences 15 min, 96 pas) → 37 GiB ; fix `sliding_window_view` + `float32` + export split par bloc.
  3. **Disque plein** sur `X_train.csv` (~20 Go partiel supprimé) ; `.npz` = format principal.
  4. **Ordre dimensions** : `.transpose(0, 2, 1)` pour obtenir `(N, time_steps, clients)`.
  5. **Seuil congestion** : calculé sur train seul (anti data leakage) ; label **global réseau** (1 label/heure), pas par nœud.
  6. **Partage équipe** : fichiers data dans `.gitignore` → Google Drive obligatoire (~1,1 Go) ; script + `requirements.txt` via Git.

* **Point d'arrêt et Prochaine étape :**
  - **Point d'arrêt** : **Phase 2 terminée à 100 %** (D, E, F, G). Script exécuté avec succès. Commande :
    ```powershell
    cd OPxGNN\dataset\dataelectricity
    python -u ..\..\scripts\01_preprocess_ld2011_2014.py
    ```
  - **Prochaine étape — Phase 3 (Membre 3, Tâches H/I/J)** : matrice Pearson sur **train uniquement** (`load_hourly.csv.gz` ou `train_data.csv`), seuil de corrélation, matrice d'adjacence, visualisation NetworkX. Puis **Phase 4** (XGBoost : `train_data.csv` + `y_cls_*.csv`) et **Phase 5** (Youssef : `sequences_*.npz` + adjacence Membre 3).



  ### 📝 Rapport de session - 2026-05-31 / Noureddine

* **Tâche(s) traitée(s) :** Phase 3 complète — **Tâche H** (matrice de corrélation de Pearson sur train uniquement), **Tâche I** (seuil de corrélation + matrice d'adjacence), **Tâche J** (visualisation du graphe avec NetworkX).

* **Ce qui a été accompli :**
  - **Environnement** : migration de Python 3.14 (instable) vers Python 3.11.9 + recréation du venv + réinstallation de toutes les dépendances + installation de `networkx` et `matplotlib`. Script `01_preprocess_ld2011_2014.py` relancé et validé avec succès (résultats identiques à la session d'Ibtihaal).
  - **Tâche H** : chargement de `load_hourly.csv.gz`, extraction des 370 colonnes clients, split train (70% = 24 545 heures, 2011-01-01 → 2013-10-19). Détection et exclusion de **22 clients à variance nulle** (apparus après 2011, consommation = 0 sur tout le train). Calcul de la matrice de corrélation de Pearson via `np.corrcoef(values.T)` sur **348 clients valides** → matrice (348, 348), min = -0.84, max = 1.00, moyenne hors diagonale = 0.37.
  - **Tâche I** : analyse de la distribution des corrélations (quantiles 25%=0.037, 50%=0.302, 75%=0.782, 85%=0.853, 90%=0.877). Simulation de densité pour seuils 0.50 à 0.90. Seuil retenu : **0.85** (équilibre densité/nœuds isolés). Matrice d'adjacence binaire résultante : **348 nœuds, 9 442 arêtes, densité 15.64%, 46 nœuds isolés**.
  - **Tâche J** : construction du graphe NetworkX depuis la matrice d'adjacence. Statistiques : degré moyen = 54.26, degré max = 126, 51 composantes connexes, composante principale = 152 nœuds. Visualisation générée : graphe spring layout coloré par degré + histogramme de distribution des degrés.

* **Fichiers créés ou modifiés :**
  - **Créé** : `OPxGNN/scripts/02_graph_topology.py`
  - **`OPxGNN/data/graph/`** :
    - `correlation_matrix.npy` (473.2 KB) — matrice Pearson float32 (348, 348)
    - `adjacency_matrix.npy` (118.4 KB) — matrice d'adjacence binaire uint8 (348, 348)
    - `adjacency_matrix.csv` (236.9 KB) — version lisible
    - `client_cols_valid.txt` (2.7 KB) — liste des 348 clients valides
    - `graph_info.csv` (0.1 KB) — paramètres : seuil=0.85, nœuds=348, arêtes=9442, densité=0.1564, isolés=46
    - `graph_visualization.png` (289.3 KB) — visualisation NetworkX

* **Problèmes rencontrés / Choix techniques :**
  1. **Python 3.14 incompatible** → `KeyboardInterrupt` lors du `pd.read_csv` ; fix : migration vers Python 3.11.9.
  2. **Calcul Pearson manuel (numpy)** → valeurs à 0.0000 (erreur de normalisation) ; fix : utilisation de `np.corrcoef(values.T)`.
  3. **22 clients à variance nulle** → `RuntimeWarning: invalid value encountered in divide` ; fix : détection par `std == 0` et exclusion avant le calcul.
  4. **Choix du seuil** : 0.85 retenu après simulation — densité 15.64%, 46 nœuds isolés ; seuil 0.90 trop sparse (84 isolés), seuil 0.80 trop dense (23%).

* **Point d'arrêt et Prochaine étape :**
  - **Point d'arrêt** : **Phase 3 terminée à 100%** (H, I, J). Script `02_graph_topology.py` exécuté et validé. Matrice d'adjacence prête dans `data/graph/adjacency_matrix.npy`.
  - **Prochaine étape immédiate — Phase 4 (Membre 2, Tâches K/L/M)** : entraînement XGBoost sur `train_data.csv` + `y_cls_train.csv`, optimisation hyperparamètres sur validation, évaluation sur test. **ET Phase 5 (Youssef, Tâches N/O/P)** : charger `sequences_*.npz` + `adjacency_matrix.npy` pour construire les objets `Data` PyTorch Geometric et coder l'architecture GCN.

