# Phase 4 : Machine Learning Classique — Baseline XGBoost

## 📋 Vue d'ensemble

Ce document décrit la Phase 4 du projet OPxGNN, qui implémente un baseline de **Machine Learning classique** (tabulaire) avec **XGBoost** pour servir de point de comparaison avec le modèle **Graph Neural Network (GCN)** de la Phase 5.

### Tâches couvertes

- **Tâche K** : Entraînement de la baseline XGBoost sur données purement tabulaires d'entraînement
- **Tâche L** : Optimisation des hyperparamètres (max_depth, learning_rate, subsample, colsample_bytree, etc.) sur l'ensemble de validation
- **Tâche M** : Évaluation du modèle sur l'ensemble de test et sauvegarde des prédictions brutes

## 🎯 Contexte scientifique et technique

### Problème
Nous disposons d'une **série temporelle multivariée** de consommation électrique (370 clients, données horaires) divisée en trois ensembles :
- **Train** : 2011-2013 (24 545 heures / 24 528 séquences de 24h)
- **Validation** : 2014 H1 (5 259 heures / 5 256 séquences)
- **Test** : 2014 H2 (5 261 heures / 5 257 séquences)

### Cible
**Classification binaire** :
- `0` = Réseau en état "Normal"
- `1` = Réseau en état "Congestionné/Critique"

La classe positive est définie comme : *charge totale > 95e centile (seuil = 360 203,93 kW)* calculé sur l'ensemble d'entraînement.

### Déséquilibre des classes
- Classe 0 (Normal)       : **33 238 / 35 065 (94,79%)**
- Classe 1 (Congestion)   : **1 827 / 35 065 (5,21%)**

### Données d'entrée pour XGBoost
Les données utilisées sont **purement tabulaires** (pas de graphe) :
- **Features** : 370 colonnes (un client par colonne, valeurs de consommation normalisées en [0, 1])
- **Shape** : 
  - Train : (24 528, 370)
  - Validation : (5 256, 370)
  - Test : (5 257, 370)

*Note* : Les 22 clients à variance nulle (apparus après 2011, consommation = 0 sur tout le train) ont déjà été exclus lors du calcul du graphe (Phase 3), mais les fichiers `train_data.csv`, etc. conservent les 370 colonnes originales pour cohérence.

## 📊 Approche XGBoost

### Pourquoi XGBoost ?
1. **Robustesse** : Excellent pour les problèmes tabulaires hétérogènes
2. **Interprétabilité** : Feature importance clairement disponible
3. **Performance** : Souvent champion des compétitions de tabular ML
4. **Baseline solide** : Permet une comparaison juste avec GNN

### Architecture du modèle

#### Tâche K : Entraînement initial
Le modèle est d'abord entraîné avec des **hyperparamètres par défaut raisonnables** :
```python
params_initial = {
    'objective': 'binary:logistic',      # Classification binaire
    'max_depth': 6,                       # Profondeur des arbres
    'learning_rate': 0.1,                 # Taux d'apprentissage
    'n_estimators': 100,                  # Nombre d'arbres initiaux
    'subsample': 0.8,                     # Fraction des samples par arbre
    'colsample_bytree': 0.8,              # Fraction des features par arbre
    'min_child_weight': 1,                # Poids minimum feuille
    'gamma': 0,                           # Complexité régularisation
    'eval_metric': 'logloss',             # Métrique de validation
    'tree_method': 'hist',                # Algorithme optimisé (rapide)
}
```

#### Tâche L : Optimisation bayésienne (Optuna)
L'optimisation des hyperparamètres utilise **Optuna** avec **Bayesian Optimization** :

**Hyperparamètres optimisés** :
| Paramètre | Plage | Raison |
|-----------|-------|--------|
| `max_depth` | [3, 12] | Contrôler la complexité et l'overfitting |
| `learning_rate` | [0.01, 0.3] | Équilibre convergence/stabilité |
| `subsample` | [0.6, 1.0] | Réduire l'overfitting (données par arbre) |
| `colsample_bytree` | [0.6, 1.0] | Réduire l'overfitting (features par arbre) |
| `min_child_weight` | [1, 7] | Régularisation (poids minimum feuille) |
| `gamma` | [0, 5] | Régularisation (coût de split) |

**Stratégie d'optimisation** :
- **Nombre d'essais** : 50 (par défaut, configurable avec `--n-trials`)
- **Fonction objectif** : Maximiser **F1-Score weighted** en validation croisée (5 folds) combiné au F1-Score sur l'ensemble de validation
- **Pruning** : Median Pruner (arrêter les trials non prometteurs tôt)
- **Sampler** : TPE (Tree-structured Parzen Estimator)

**Validation croisée** :
- La validation croisée est effectuée sur l'ensemble **d'entraînement uniquement** (5 folds)
- L'ensemble de **validation** est utilisé pour évaluer les hyperparamètres à chaque trial
- Cela prévient la fuite d'information (data leakage)

#### Tâche M : Évaluation finale
Une fois les hyperparamètres optimisés, le modèle est évalué sur l'ensemble de **test** non vu durant l'entraînement/optimisation.

**Métriques d'évaluation** :
- **Accuracy** : $(TP + TN) / (TP + TN + FP + FN)$
- **Precision** : $TP / (TP + FP)$ — "Quand on prédit congestion, on a raison combien de fois ?"
- **Recall (Sensitivity)** : $TP / (TP + FN)$ — "Combien de vrais cas de congestion on détecte ?"
- **Specificity** : $TN / (TN + FP)$ — "Combien de vrais cas normaux on détecte ?"
- **F1-Score** : Moyenne harmonique (Precision, Recall) — Important avec données déséquilibrées
- **ROC-AUC** : Aire sous la courbe ROC — Performance globale

## 🚀 Comment lancer le script

### Prérequis
1. **Python 3.11+** installé et disponible dans le PATH
2. **Les données prétraitées** doivent exister dans `dataset/dataelectricity/` :
   - `train_data.csv`
   - `validation_data.csv`
   - `test_data.csv`
   - `y_cls_train.csv`, `y_cls_val.csv`, `y_cls_test.csv`

### Étape 1 : Créer l'environnement virtuel et installer les dépendances

#### Option 1 : Via le script batch (Windows CMD)
```batch
cd C:\Users\lenovo\Desktop\PROJETSDD\OPxGNN
run_xgboost.bat
```

#### Option 2 : Via PowerShell
```powershell
cd C:\Users\lenovo\Desktop\PROJETSDD\OPxGNN
.\run_xgboost.ps1
```

#### Option 3 : Manuel
```powershell
cd C:\Users\lenovo\Desktop\PROJETSDD\OPxGNN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install optuna  # Optionnel mais recommandé
python scripts/03_baseline_xgboost.py
```

### Étape 2 : Options de lancement

#### Lancement standard (avec optimisation)
```bash
python scripts/03_baseline_xgboost.py
```

Cela exécutera :
- **Tâche K** : Entraînement initial
- **Tâche L** : Optimisation Optuna (50 essais, 5 folds CV)
- **Tâche M** : Évaluation sur test

#### Lancement sans optimisation (debug)
```bash
python scripts/03_baseline_xgboost.py --skip-optimization
```

Utile pour déboguer ou faire des tests rapides.

#### Options avancées
```bash
python scripts/03_baseline_xgboost.py --n-trials 100 --cv-folds 10 --seed 123
```

- `--n-trials N` : Nombre d'essais Optuna (défaut : 50)
- `--cv-folds N` : Nombre de folds validation croisée (défaut : 5)
- `--seed SEED` : Graine aléatoire pour reproductibilité (défaut : 42)
- `--skip-optimization` : Ignorer Optuna, utiliser paramètres prédéfinis

## 📁 Fichiers générés

Le script produit les **artefacts suivants** dans `data/processed/` :

| Fichier | Description |
|---------|-------------|
| `xgboost_baseline_model.pkl` | Modèle XGBoost entraîné et sérialisé |
| `xgboost_predictions_train.csv` | Prédictions (y_true, y_pred, y_proba) sur train |
| `xgboost_predictions_val.csv` | Prédictions sur validation |
| `xgboost_predictions_test.csv` | Prédictions sur test |
| `xgboost_probabilities_test.csv` | Probabilités détaillées (prob_normal, prob_congestion) sur test |
| `xgboost_hyperparameters.json` | Hyperparamètres optimaux trouvés |
| `xgboost_evaluation_metrics.json` | Métriques complètes (train, val, test, logs d'optim) |
| `xgboost_feature_importance.csv` | Importance des 370 features tabulaires |

## 📊 Résultats attendus

### Performance de baseline
Pour un dataset déséquilibré à 5 % de classe positive, attendez typiquement :
- **Accuracy** : 94-97% (mais attention au biais classe dominante)
- **Precision** : 50-80% (si on prédot congestion, c'est souvent correct)
- **Recall** : 40-70% (on détecte 40-70% des vraies congestions)
- **F1-Score** : 0.50-0.75 (important avec déséquilibre)
- **ROC-AUC** : 0.80-0.95 (assez bon discriminant)

### Importance des features
XGBoost révélera quels **clients** (features) sont les plus importants pour prédire la congestion. Cela peut guider l'analyse :
- Les clients avec forte consommation/variance auront forte importance
- Les clients "redondants" auront faible importance

## 🔄 Prochaines étapes

Après la Phase 4, on peut :
1. **Phase 5** (Youssef) : Coder l'architecture GCN et entraîner le modèle de graphe
2. **Phase 7** (Noureddine) : Comparer XGBoost vs GCN, calculer le "gain" de la structure graphe

## 📝 Glossaire

| Terme | Définition |
|-------|-----------|
| **XGBoost** | eXtreme Gradient Boosting — algorithme d'ensemble itératif |
| **Hyperparamètres** | Variables contrôlant l'apprentissage (non appris par les données) |
| **Validation croisée** | Division répétée du train en under-folds pour estimer la généralisation |
| **Optuna** | Framework Python d'optimisation bayésienne des hyperparamètres |
| **Overfitting** | Quand le modèle apprend le bruit, test perf << train perf |
| **ROC-AUC** | Métrique robuste au déséquilibre de classes |

## 🐛 Dépannage

### Erreur : "Python introuvable"
→ Installer Python 3.11+ depuis python.org ou via Windows Package Manager
→ Ajouter Python au PATH dans les Variables d'environnement Windows

### Erreur : "Module optuna introuvable"
→ Installer manuellement : `pip install optuna`
→ Ou relancer le script batch qui l'installe automatiquement

### Erreur : "Fichiers de données introuvables"
→ S'assurer que le script `01_preprocess_ld2011_2014.py` a été exécuté complètement
→ Vérifier que les fichiers existent dans `dataset/dataelectricity/`

### RAM insuffisante durant optimisation
→ Réduire `--n-trials` ou `--cv-folds`
→ Ou utiliser `--skip-optimization` pour un test rapide

## 📚 Références

- XGBoost Documentation : https://xgboost.readthedocs.io/
- Optuna Documentation : https://optuna.readthedocs.io/
- Scikit-learn Metrics : https://scikit-learn.org/stable/modules/model_evaluation.html
- Class Imbalance in ML : https://developers.google.com/machine-learning/glossary#class-imbalanced-dataset

---

**Auteur** : Team IA / Data Science  
**Date** : 2026-06-06  
**Langue** : Français (commentaires et documentation)
