# 🔧 Instructions d'installation et de configuration — Phase 4 XGBoost

## ⚠️ État actuel
L'installation de Python 3.11 via Windows Package Manager (`winget`) est **en cours**. 
Cela peut prendre quelques minutes.

## ✅ Étapes après l'installation de Python

Une fois que Python est installé, veuillez suivre ces étapes :

### 1. **Fermer et rouvrir PowerShell**
   - Cela permettra à PowerShell de reconnaître Python dans le PATH
   - Utilisez une **nouvelle fenêtre PowerShell** (ne pas réutiliser la fenêtre précédente)

### 2. **Vérifier que Python est accessible**
   ```powershell
   python --version
   ```
   
   Vous devriez voir : `Python 3.11.9` (ou version similaire)
   
   Si vous voyez toujours "Python introuvable", consultez la section **Dépannage** ci-dessous.

### 3. **Naviguer au répertoire du projet**
   ```powershell
   cd C:\Users\lenovo\Desktop\PROJETSDD\OPxGNN
   ```

### 4. **Lancer l'entraînement XGBoost**
   
   #### Option A : Via le script batch
   ```batch
   run_xgboost.bat
   ```
   
   #### Option B : Via PowerShell
   ```powershell
   .\run_xgboost.ps1
   ```
   
   #### Option C : Manuellement
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install optuna
   python scripts/03_baseline_xgboost.py
   ```

## 🐛 Dépannage

### Problème : "Python introuvable" ou ouverture du Microsoft Store
   
   **Solution 1** : Désactiver l'alias App Execution dans Windows Settings
   1. Ouvrez **Settings** → **Apps** → **App execution aliases**
   2. Désactivez **python.exe** et **python3.exe** (les toggles)
   3. Rouvrez PowerShell et réessayez
   
   **Solution 2** : Utiliser le chemin complet de Python
   ```powershell
   & "C:\Program Files\Python311\python.exe" -m venv .venv
   ```
   
   **Solution 3** : Réinstaller Python
   ```powershell
   winget uninstall Python.Python.3.11
   winget install Python.Python.3.11 -e --silent
   ```
   Puis redémarrer l'ordinateur.

### Problème : "Impossible de trouver le module pip"
   
   **Solution** : Mettre à jour Python
   ```powershell
   python -m pip install --upgrade pip
   ```

### Problème : "Dépendances manquantes (numpy, pandas, xgboost, etc.)"
   
   **Solution** : Installer depuis `requirements.txt`
   ```powershell
   pip install -r requirements.txt
   ```

### Problème : "Optuna non trouvé" (avertissement non bloquant)
   
   **Solution** : Installer manuellement
   ```powershell
   pip install optuna
   ```

### Problème : Erreur de disque plein ou RAM insuffisante
   
   **Solution** : Lancer sans optimisation Optuna
   ```powershell
   python scripts/03_baseline_xgboost.py --skip-optimization
   ```
   
   Ou réduire les essais :
   ```powershell
   python scripts/03_baseline_xgboost.py --n-trials 10 --cv-folds 3
   ```

## 📊 Temps d'exécution attendu

- **Sans optimisation** : ~5-10 minutes
- **Avec optimisation** (50 essais, 5 folds CV) : **30-60 minutes** (dépend du CPU)

## 📁 Fichiers d'aide

Consultez les fichiers suivants pour plus de détails :

- [PHASE4_XGBOOST.md](PHASE4_XGBOOST.md) — Documentation complète de la Phase 4
- [scripts/03_baseline_xgboost.py](scripts/03_baseline_xgboost.py) — Code source du script
- [run_xgboost.ps1](run_xgboost.ps1) — Script PowerShell de lancement
- [run_xgboost.bat](run_xgboost.bat) — Script batch de lancement

## ✨ Prochaines étapes après succès

Une fois que le script s'exécute avec succès et produit les fichiers dans `data/processed/`, vous pouvez :

1. **Consulter les résultats** :
   ```powershell
   Get-Content data/processed/xgboost_evaluation_metrics.json | ConvertFrom-Json | Format-Table
   ```

2. **Passer à la Phase 5** (Youssef) : Entraîner le modèle GCN

3. **Comparer les modèles** (Phase 7, Noureddine) : XGBoost vs GCN

---

**Dernière mise à jour** : 2026-06-06  
**Langue** : Français
