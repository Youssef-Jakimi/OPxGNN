#!/usr/bin/env pwsh
<#
    ============================================================================
    Script de lancement pour la Phase 4 : Baseline XGBoost (PowerShell)
    ============================================================================
    Ce script active l'environnement virtuel et lance le script XGBoost
    
    Utilisation:
        .\run_xgboost.ps1
        ou
        .\run_xgboost.ps1 -SkipOptimization
    ============================================================================
#>

param(
    [Switch] $SkipOptimization = $False
)

# Vérifier si nous sommes au bon endroit
if (-not (Test-Path "scripts\03_baseline_xgboost.py")) {
    Write-Host "[ERREUR] Fichier scripts\03_baseline_xgboost.py introuvable" -ForegroundColor Red
    Write-Host "Assurez-vous d'exécuter ce script depuis le répertoire racine du projet" -ForegroundColor Red
    exit 1
}

# Créer l'environnement virtuel s'il n'existe pas
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Création de l'environnement virtuel..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] Impossible de créer l'environnement virtuel" -ForegroundColor Red
        exit 1
    }
}

# Activer l'environnement virtuel
Write-Host "[INFO] Activation de l'environnement virtuel..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Mettre à jour pip
Write-Host "[INFO] Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install -U pip setuptools wheel --quiet

# Installer les dépendances
Write-Host "[INFO] Installation des dépendances..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERREUR] Impossible d'installer les dépendances" -ForegroundColor Red
    exit 1
}

# Installer Optuna (optionnel mais recommandé)
pip install optuna --quiet

# Construire les arguments pour le script Python
$args_list = @("scripts\03_baseline_xgboost.py")
if ($SkipOptimization) {
    $args_list += "--skip-optimization"
}

# Lancer le script XGBoost
Write-Host "[INFO] Lancement du script de baseline XGBoost..." -ForegroundColor Yellow
Write-Host ""
python @args_list

# Gérer les erreurs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERREUR] Une erreur s'est produite lors de l'exécution" -ForegroundColor Red
    exit 1
}
