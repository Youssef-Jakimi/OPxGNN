@echo off
REM ============================================================================
REM Script de lancement pour la Phase 4 : Baseline XGBoost
REM ============================================================================
REM Ce script active l'environnement virtuel et lance le script XGBoost
REM Utilisation: 
REM   run_xgboost.bat
REM   ou
REM   run_xgboost.bat --skip-optimization
REM ============================================================================

setlocal enabledelayedexpansion

REM Vérifier si nous sommes au bon endroit
if not exist "scripts\03_baseline_xgboost.py" (
    echo [ERREUR] Fichier scripts\03_baseline_xgboost.py introuvable
    echo Assurez-vous d'exécuter ce script depuis le répertoire racine du projet
    pause
    exit /b 1
)

REM Créer l'environnement virtuel s'il n'existe pas
if not exist ".venv" (
    echo [INFO] Création de l'environnement virtuel...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERREUR] Impossible de créer l'environnement virtuel
        pause
        exit /b 1
    )
)

REM Activer l'environnement virtuel
echo [INFO] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

REM Mettre à jour pip
echo [INFO] Mise à jour de pip...
python -m pip install -U pip setuptools wheel --quiet

REM Installer les dépendances
echo [INFO] Installation des dépendances...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERREUR] Impossible d'installer les dépendances
    pause
    exit /b 1
)

REM Installer Optuna (optionnel mais recommandé)
pip install optuna --quiet

REM Lancer le script XGBoost
echo [INFO] Lancement du script de baseline XGBoost...
echo.
python scripts\03_baseline_xgboost.py %*

REM Garder la fenêtre ouverte si une erreur s'est produite
if errorlevel 1 (
    echo.
    echo [ERREUR] Une erreur s'est produite lors de l'exécution
    pause
)

endlocal
