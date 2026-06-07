#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Phase 4 : Machine Learning Classique (Baseline Tabulaire) — XGBoost
================================================================================

Tâche K : Entraîner la baseline XGBoost sur données purement tabulaires.
Tâche L : Optimiser les hyperparamètres (profondeur, learning rate).
Tâche M : Évaluer sur ensemble de Test et sauvegarder les prédictions.

Entrées:
  - dataset/dataelectricity/train_data.csv : caractéristiques d'entraînement (24 545 × 370)
  - dataset/dataelectricity/validation_data.csv : caractéristiques de validation (5 259 × 370)
  - dataset/dataelectricity/test_data.csv : caractéristiques de test (5 261 × 370)
  - dataset/dataelectricity/y_cls_train.csv : labels d'entraînement (0/1)
  - dataset/dataelectricity/y_cls_val.csv : labels de validation (0/1)
  - dataset/dataelectricity/y_cls_test.csv : labels de test (0/1)

Sorties:
  - data/processed/xgboost_baseline_model.pkl : modèle XGBoost entraîné et optimisé
  - data/processed/xgboost_predictions_train.csv : prédictions brutes sur train
  - data/processed/xgboost_predictions_val.csv : prédictions brutes sur validation
  - data/processed/xgboost_predictions_test.csv : prédictions brutes sur test
  - data/processed/xgboost_probabilities_test.csv : probabilités de classe sur test
  - data/processed/xgboost_hyperparameters.json : hyperparamètres optimaux trouvés
  - data/processed/xgboost_evaluation_metrics.json : métriques d'évaluation (Acc, Prec, Rec, F1)

Exécution:
  python scripts/03_baseline_xgboost.py [OPTIONS]

OPTIONS:
  --skip-optimization   Ignorer l'optimisation des hyperparamètres
  --cv-folds N          Nombre de folds pour la validation croisée (défaut: 5)
  --n-trials N          Nombre d'essais pour l'optimisation Optuna (défaut: 50)
  --seed SEED           Graine aléatoire pour reproductibilité (défaut: 42)
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Optuna pour optimisation bayésienne des hyperparamètres
try:
    import optuna
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[⚠ AVERTISSEMENT] Optuna non installé. Installation recommandée : pip install optuna")

warnings.filterwarnings('ignore')


class XGBoostBaseline:
    """
    Classe pour entraîner, optimiser et évaluer un modèle XGBoost comme baseline tabulaire.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialise la classe.
        
        Args:
            seed: Graine aléatoire pour reproductibilité
        """
        self.seed = seed
        self.model = None
        self.best_params = None
        self.history = {
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {},
            'optimization_log': []
        }
        
    def charger_donnees(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray, np.ndarray]:
        """
        Charge les données d'entraînement, validation et test.
        
        Args:
            data_dir: Répertoire contenant les fichiers de données
            
        Returns:
            Tuple (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("[📂 CHARGEMENT] Chargement des données...")
        
        # Charger les fichiers CSV complets
        train_full = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
        val_full = pd.read_csv(os.path.join(data_dir, 'validation_data.csv'))
        test_full = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
        
        # Extraire X (colonnes MT_001 à MT_370 uniquement) et y (label)
        mt_cols = [col for col in train_full.columns if col.startswith('MT_')]
        
        X_train = train_full[mt_cols].copy()
        y_train = train_full['label'].copy() if 'label' in train_full.columns else pd.read_csv(os.path.join(data_dir, 'y_cls_train.csv'))
        
        X_val = val_full[mt_cols].copy()
        y_val = val_full['label'].copy() if 'label' in val_full.columns else pd.read_csv(os.path.join(data_dir, 'y_cls_val.csv'))
        
        X_test = test_full[mt_cols].copy()
        y_test = test_full['label'].copy() if 'label' in test_full.columns else pd.read_csv(os.path.join(data_dir, 'y_cls_test.csv'))
        
        # Convertir les colonnes non numériques en valeurs numériques exploitables par XGBoost
        for df in [X_train, X_val, X_test]:
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                try:
                    # Tenter d'abord une conversion vers datetime puis vers timestamp
                    df[col] = pd.to_datetime(df[col], errors='raise').astype('int64') // 10**9
                except (ValueError, TypeError):
                    # Sinon, factoriser les valeurs catégorielles arbitrairement
                    df[col], _ = pd.factorize(df[col])

        # Aligner les formes X / y en cas de label manquant ou de découpage différent
        if len(X_train) != len(y_train):
            min_len = min(len(X_train), len(y_train))
            print(f"[⚠] Alignement Train : réduction de X_train de {len(X_train) - min_len} lignes")
            X_train = X_train.iloc[:min_len, :]
            y_train = y_train.iloc[:min_len] if isinstance(y_train, pd.Series) else y_train[:min_len]
        if len(X_val) != len(y_val):
            min_len = min(len(X_val), len(y_val))
            print(f"[⚠] Alignement Validation : réduction de X_val de {len(X_val) - min_len} lignes")
            X_val = X_val.iloc[:min_len, :]
            y_val = y_val.iloc[:min_len] if isinstance(y_val, pd.Series) else y_val[:min_len]
        if len(X_test) != len(y_test):
            min_len = min(len(X_test), len(y_test))
            print(f"[⚠] Alignement Test : réduction de X_test de {len(X_test) - min_len} lignes")
            X_test = X_test.iloc[:min_len, :]
            y_test = y_test.iloc[:min_len] if isinstance(y_test, pd.Series) else y_test[:min_len]

        # Extraire les valeurs et aplatir si nécessaire
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values
        
        # Extraire y (gérer pandas Series et DataFrames)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0].values.ravel()
        elif isinstance(y_train, pd.Series):
            y_train = y_train.values.ravel()
        
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.iloc[:, 0].values.ravel()
        elif isinstance(y_val, pd.Series):
            y_val = y_val.values.ravel()
            
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0].values.ravel()
        elif isinstance(y_test, pd.Series):
            y_test = y_test.values.ravel()
        
        print(f"[✓] Données chargées :")
        print(f"    • Train    : X={X_train.shape}, y={y_train.shape}, Congestion={y_train.sum()}/{len(y_train)} ({100*y_train.mean():.2f}%)")
        print(f"    • Validation : X={X_val.shape}, y={y_val.shape}, Congestion={y_val.sum()}/{len(y_val)} ({100*y_val.mean():.2f}%)")
        print(f"    • Test     : X={X_test.shape}, y={y_test.shape}, Congestion={y_test.sum()}/{len(y_test)} ({100*y_test.mean():.2f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def entrainer_baseline(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne un modèle XGBoost initial (avant optimisation).
        
        Tâche K : Entraînement basique avec paramètres par défaut.
        
        Args:
            X_train: Caractéristiques d'entraînement
            y_train: Labels d'entraînement
        """
        print("\n[🚀 TÂCHE K] Entraînement du modèle XGBoost baseline...")
        
        # Hyperparamètres initiaux raisonnables
        params_initial = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': self.seed,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cpu'
        }
        
        # Initialiser le modèle
        self.model = xgb.XGBClassifier(**params_initial)
        
        # Entraîner
        self.model.fit(X_train, y_train)
        
        # Évaluer sur train
        y_pred_train = self.model.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train, zero_division=0)
        
        self.history['train_metrics'] = {
            'accuracy': float(acc_train),
            'f1': float(f1_train)
        }
        
        print(f"[✓] Baseline entraîné")
        print(f"    • Accuracy (train) : {acc_train:.4f}")
        print(f"    • F1-Score (train) : {f1_train:.4f}")
    
    def optimiser_hyperparametres(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  n_trials: int = 50, cv_folds: int = 5) -> None:
        """
        Optimise les hyperparamètres de XGBoost via validation sur ensemble de validation.
        
        Tâche L : Optimisation de max_depth, learning_rate, subsample, colsample_bytree.
        
        Args:
            X_train: Caractéristiques d'entraînement
            y_train: Labels d'entraînement
            X_val: Caractéristiques de validation
            y_val: Labels de validation
            n_trials: Nombre d'essais d'optimisation
            cv_folds: Nombre de folds pour la validation croisée
        """
        print("\n[🔧 TÂCHE L] Optimisation des hyperparamètres XGBoost...")
        
        if not OPTUNA_AVAILABLE:
            print("[⚠ INFO] Optuna non disponible. Utilisation de paramètres prédéfinis optimisés.")
            self.best_params = {
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.85,
                'min_child_weight': 1,
                'gamma': 1,
                'n_estimators': 200
            }
            print(f"[✓] Paramètres prédéfinis appliqués : {self.best_params}")
            self._retrainer_avec_params(X_train, y_train, X_val, y_val)
            return
        
        # Définir l'objectif d'optimisation Optuna
        def objective_fn(trial):
            """
            Fonction objective pour Optuna : maximiser F1-Score sur validation.
            """
            # Suggérer les hyperparamètres
            max_depth = trial.suggest_int('max_depth', 3, 12)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 7)
            gamma = trial.suggest_float('gamma', 0, 5)
            
            params = {
                'objective': 'binary:logistic',
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': 150,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'random_state': self.seed,
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'device': 'cpu',
                'verbosity': 0
            }
            
            # Entraîner le modèle
            xgb_model = xgb.XGBClassifier(**params)
            
            # Évaluation via validation croisée sur train
            cv_scores = cross_val_score(xgb_model, X_train, y_train, 
                                       cv=cv_folds, scoring='f1_weighted', n_jobs=-1)
            
            # Validation finale sur ensemble de validation
            xgb_model.fit(X_train, y_train)
            y_pred_val = xgb_model.predict(X_val)
            f1_val = f1_score(y_val, y_pred_val, zero_division=0)
            
            # Métriques combinées (moyenne CV + F1 validation)
            score = (cv_scores.mean() + f1_val) / 2.0
            
            # Logging
            self.history['optimization_log'].append({
                'trial': trial.number,
                'params': params,
                'cv_f1': float(cv_scores.mean()),
                'val_f1': float(f1_val),
                'score': float(score)
            })
            
            if (trial.number + 1) % 10 == 0:
                print(f"    Trial {trial.number + 1}/{n_trials} : F1-Score={score:.4f}")
            
            return score
        
        # Créer l'étude Optuna
        print(f"    Lancement de {n_trials} essais d'optimisation...")
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = MedianPruner()
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
        
        # Récupérer les meilleurs paramètres
        best_trial = study.best_trial
        self.best_params = {
            'max_depth': best_trial.params['max_depth'],
            'learning_rate': best_trial.params['learning_rate'],
            'subsample': best_trial.params['subsample'],
            'colsample_bytree': best_trial.params['colsample_bytree'],
            'min_child_weight': best_trial.params['min_child_weight'],
            'gamma': best_trial.params['gamma'],
            'n_estimators': 200  # Augmenter le nombre d'arbres avec les LR optimisés
        }
        
        print(f"[✓] Hyperparamètres optimaux trouvés :")
        for param, valeur in self.best_params.items():
            print(f"    • {param} : {valeur}")
        print(f"    Meilleur F1-Score : {best_trial.value:.4f}")
        
        # Réentraîner avec les meilleurs paramètres
        self._retrainer_avec_params(X_train, y_train, X_val, y_val)
    
    def _retrainer_avec_params(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Réentraîne le modèle avec les paramètres optimisés.
        
        Args:
            X_train: Caractéristiques d'entraînement
            y_train: Labels d'entraînement
            X_val: Caractéristiques de validation
            y_val: Labels de validation
        """
        params_final = {
            'objective': 'binary:logistic',
            'random_state': self.seed,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cpu',
            'verbosity': 0,
            **self.best_params
        }
        
        self.model = xgb.XGBClassifier(**params_final)
        self.model.fit(X_train, y_train)
        
        # Évaluation sur train et validation
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        self.history['train_metrics'] = {
            'accuracy': float(accuracy_score(y_train, y_pred_train)),
            'precision': float(precision_score(y_train, y_pred_train, zero_division=0)),
            'recall': float(recall_score(y_train, y_pred_train, zero_division=0)),
            'f1': float(f1_score(y_train, y_pred_train, zero_division=0))
        }
        
        self.history['val_metrics'] = {
            'accuracy': float(accuracy_score(y_val, y_pred_val)),
            'precision': float(precision_score(y_val, y_pred_val, zero_division=0)),
            'recall': float(recall_score(y_val, y_pred_val, zero_division=0)),
            'f1': float(f1_score(y_val, y_pred_val, zero_division=0))
        }
        
        print(f"[✓] Modèle réentraîné avec paramètres optimisés")
        print(f"    Métriques (Validation) : Acc={self.history['val_metrics']['accuracy']:.4f}, "
              f"F1={self.history['val_metrics']['f1']:.4f}")
    
    def evaluer_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Évalue le modèle optimisé sur l'ensemble de test.
        
        Tâche M : Évaluation complète et génération des métriques.
        
        Args:
            X_test: Caractéristiques de test
            y_test: Labels de test
            
        Returns:
            Tuple (y_pred_test, y_proba_test)
        """
        print("\n[📊 TÂCHE M] Évaluation du modèle sur ensemble de Test...")
        
        # Prédictions
        y_pred_test = self.model.predict(X_test)
        y_proba_test = self.model.predict_proba(X_test)[:, 1]  # Probabilité de classe 1
        
        # Métriques
        acc_test = accuracy_score(y_test, y_pred_test)
        prec_test = precision_score(y_test, y_pred_test, zero_division=0)
        rec_test = recall_score(y_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba_test)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        self.history['test_metrics'] = {
            'accuracy': float(acc_test),
            'precision': float(prec_test),
            'recall': float(rec_test),
            'f1': float(f1_test),
            'roc_auc': float(roc_auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        print(f"[✓] Évaluation complétée :")
        print(f"    • Accuracy     : {acc_test:.4f}")
        print(f"    • Precision    : {prec_test:.4f}")
        print(f"    • Recall       : {rec_test:.4f}")
        print(f"    • F1-Score     : {f1_test:.4f}")
        print(f"    • ROC-AUC      : {roc_auc:.4f}")
        print(f"    • Sensitivity  : {sensitivity:.4f}")
        print(f"    • Specificity  : {specificity:.4f}")
        print(f"\n    Matrice de confusion :")
        print(f"        TN={tn}  FP={fp}")
        print(f"        FN={fn}  TP={tp}")
        
        return y_pred_test, y_proba_test
    
    def sauvegarder_sorties(self, output_dir: str, 
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           y_pred_test: np.ndarray, y_proba_test: np.ndarray) -> None:
        """
        Sauvegarde le modèle, les prédictions et les hyperparamètres optimisés.
        
        Args:
            output_dir: Répertoire de sortie
            X_train, y_train: Données d'entraînement
            X_val, y_val: Données de validation
            X_test, y_test: Données de test
            y_pred_test: Prédictions sur test
            y_proba_test: Probabilités sur test
        """
        print("\n[💾 SAUVEGARDE] Archivage des artefacts...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Sauvegarder le modèle entraîné
        model_path = os.path.join(output_dir, 'xgboost_baseline_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"    ✓ Modèle sauvegardé : {model_path}")
        
        # 2. Sauvegarder les prédictions (train)
        y_pred_train = self.model.predict(X_train)
        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        df_pred_train = pd.DataFrame({
            'y_true': y_train,
            'y_pred': y_pred_train,
            'y_proba': y_proba_train
        })
        train_pred_path = os.path.join(output_dir, 'xgboost_predictions_train.csv')
        df_pred_train.to_csv(train_pred_path, index=False)
        print(f"    ✓ Prédictions train : {train_pred_path}")
        
        # 3. Sauvegarder les prédictions (validation)
        y_pred_val = self.model.predict(X_val)
        y_proba_val = self.model.predict_proba(X_val)[:, 1]
        df_pred_val = pd.DataFrame({
            'y_true': y_val,
            'y_pred': y_pred_val,
            'y_proba': y_proba_val
        })
        val_pred_path = os.path.join(output_dir, 'xgboost_predictions_val.csv')
        df_pred_val.to_csv(val_pred_path, index=False)
        print(f"    ✓ Prédictions validation : {val_pred_path}")
        
        # 4. Sauvegarder les prédictions (test)
        df_pred_test = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test
        })
        test_pred_path = os.path.join(output_dir, 'xgboost_predictions_test.csv')
        df_pred_test.to_csv(test_pred_path, index=False)
        print(f"    ✓ Prédictions test : {test_pred_path}")
        
        # 5. Sauvegarder les probabilités (test détaillées)
        proba_path = os.path.join(output_dir, 'xgboost_probabilities_test.csv')
        df_proba = pd.DataFrame(self.model.predict_proba(X_test), 
                                columns=['prob_normal', 'prob_congestion'])
        df_proba.to_csv(proba_path, index=False)
        print(f"    ✓ Probabilités (test) : {proba_path}")
        
        # 6. Sauvegarder les hyperparamètres optimisés
        params_path = os.path.join(output_dir, 'xgboost_hyperparameters.json')
        with open(params_path, 'w') as f:
            json.dump(self.best_params or {}, f, indent=2)
        print(f"    ✓ Hyperparamètres : {params_path}")
        
        # 7. Sauvegarder les métriques d'évaluation
        metrics_path = os.path.join(output_dir, 'xgboost_evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"    ✓ Métriques d'évaluation : {metrics_path}")
        
        # 8. Sauvegarder l'importance des features
        importance_path = os.path.join(output_dir, 'xgboost_feature_importance.csv')
        feature_importance = pd.DataFrame({
            'feature': [f'Client_{i}' for i in range(X_train.shape[1])],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv(importance_path, index=False)
        print(f"    ✓ Importance des features : {importance_path}")
        
        print(f"\n[✓ SUCCÈS] Tous les artefacts ont été sauvegardés dans {output_dir}")


def main():
    """
    Fonction principale : orchestrer l'entraînement et l'évaluation XGBoost.
    """
    parser = argparse.ArgumentParser(description='Baseline XGBoost - Phase 4')
    parser.add_argument('--data-dir', type=str, 
                       default='dataset/dataelectricity',
                       help='Répertoire contenant les données')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed',
                       help='Répertoire de sortie pour les artefacts')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Ignorer l\'optimisation Optuna')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Nombre d\'essais pour Optuna')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Nombre de folds pour validation croisée')
    parser.add_argument('--seed', type=int, default=42,
                       help='Graine aléatoire')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  Phase 4 : Machine Learning Classique (Baseline Tabulaire) — XGBoost")
    print("  Tâches K, L, M : Entraînement, Optimisation, Évaluation")
    print("=" * 80 + "\n")
    
    # Initialiser
    baseline = XGBoostBaseline(seed=args.seed)
    
    # Charger les données
    X_train, y_train, X_val, y_val, X_test, y_test = baseline.charger_donnees(args.data_dir)
    
    # Tâche K : Entraîner le modèle initial
    baseline.entrainer_baseline(X_train, y_train)
    
    # Tâche L : Optimiser les hyperparamètres
    if not args.skip_optimization:
        baseline.optimiser_hyperparametres(X_train, y_train, X_val, y_val,
                                          n_trials=args.n_trials, 
                                          cv_folds=args.cv_folds)
    else:
        print("\n[ℹ INFO] Optimisation ignorée (--skip-optimization)")
    
    # Tâche M : Évaluer sur test
    y_pred_test, y_proba_test = baseline.evaluer_test(X_test, y_test)
    
    # Sauvegarder tous les artefacts
    baseline.sauvegarder_sorties(args.output_dir,
                                 X_train, y_train,
                                 X_val, y_val,
                                 X_test, y_test,
                                 y_pred_test, y_proba_test)
    
    print("\n" + "=" * 80)
    print("  ✅ Phase 4 TERMINÉE — Baseline XGBoost prêt pour comparaison avec GNN")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
