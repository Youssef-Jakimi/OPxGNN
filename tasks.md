## Starting Prompt


**[CONTEXTE GLOBAL DU PROJET]**
(Coller ici le contenu de project_context.md)

**[ÉTAT D'AVANCEMENT DE L'ÉQUIPE]**
(Coller ici le dernier contenu du fichier team_progress_summary.md. Si c'est le premier jour, écrire simplement "Début du projet, aucune tâche encore effectuée.")

**[MA TÂCHE ACTUELLE]**
Je suis responsable de la tâche suivante : [INSERER LA LETTRE ET LA DESCRIPTION DE LA TÂCHE ICI, ex: Tâche E : Rééchantillonner les séries temporelles de 15 minutes à 1 heure...]

**[INSTRUCTIONS POUR L'IA]**
Tu es mon assistant technique senior (Data Scientist / Ingénieur IA). Ton but est de m'aider à accomplir cette tâche spécifique de A à Z.
1. Lis le contexte global et l'état d'avancement pour comprendre exactement où nous en sommes dans le projet.
2. Commence par m'expliquer en détail ce que nous devons faire pour MA tâche et pourquoi c'est important mathématiquement ou techniquement pour cette étape.
3. Propose-moi une première ébauche de code Python (ou la marche à suivre si ce n'est pas du code).
4. Attends mes retours. Ne fais pas tout d'un coup. Pose-moi des questions si des choix techniques se présentent, et guidons-nous étape par étape jusqu'à ce que la tâche soit testée et validée.
5. Règle absolue : Toutes nos interactions, explications, et commentaires de code doivent être en français.

## Summary Prompt

La session de travail d'aujourd'hui est terminée. Nous avons fini (ou bien avancé sur) la tâche qui m'était assignée.

**[INSTRUCTIONS POUR L'IA]**
Génère un résumé clair, concis et très technique de ce que nous venons d'accomplir dans cette conversation. Ce résumé sera lu par mes collaborateurs humains ET par leurs propres agents IA demain pour qu'ils puissent reprendre le relais exactement là où nous nous sommes arrêtés.

Formate ta réponse strictement en Markdown avec la structure suivante :

### 📝 Rapport de session - [Insérer la Date] / [Insérer mon Prénom]
* **Tâche(s) traitée(s) :** (Rappel de la lettre et de la tâche)
* **Ce qui a été accompli :** (Résumé technique détaillé de ce que le code fait désormais, ex: "Dataset nettoyé, seuil de 95% calculé à 2500 kW, labels 0 et 1 générés...")
* **Fichiers créés ou modifiés :** (ex: `preprocessing.py`, `cleaned_data.csv`)
* **Problèmes rencontrés / Choix techniques :** (Mentionne ici les bugs résolus ou les choix d'architecture que nous avons pris)
* **Point d'arrêt et Prochaine étape :** (Où je me suis arrêté avec précision, et quelle est l'étape suivante immédiate pour l'équipe)

Ne génère QUE ce texte Markdown. N'ajoute pas d'introduction (du type "Voici votre résumé") ni de conclusion. Je dois pouvoir copier-coller ta réponse directement dans notre fichier de suivi commun.

# #####################################################################################

# Phase 1 : Initialisation et Configuration (Setup)
• Tâche A : Créer le dépôt Git (ex: sur GitHub), configurer le fichier .gitignore (pour ignorer les gros fichiers CSV/TXT) et initialiser le requirements.txt. (Assigné à : Youssef - Architecte IA)
• Tâche B : Télécharger le dataset complet, créer un dossier data/ en local, et partager le fichier brut sécurisé (ex: via Google Drive) avec l'équipe. (Assigné à : Membre 2 - Spécialiste Data)
• Tâche C : Initialiser le squelette du rapport final en LaTeX sur Overleaf ou en local pour que l'équipe puisse y documenter ses avancées. (Assigné à : Membre 3 - Analyste Graphes)


# Phase 2 : Prétraitement des Données (Data Preprocessing)
• Tâche D : Coder le script d'importation pandas (gérer les séparateurs ; et les décimales ,). (Assigné à : Membre 2)
• Tâche E : Rééchantillonner les séries temporelles de 15 minutes à 1 heure pour lisser les anomalies temporelles et combler les valeurs nulles (fillna). (Assigné à : Membre 2)
• Tâche F : Créer la variable cible de classification (Target) en calculant la charge totale et en appliquant un seuil (ex: 95ème centile) pour obtenir des labels 0 (Normal) et 1 (Congestionné). (Assigné à : Membre 2)
• Tâche G : Diviser le dataset de manière strictement chronologique (Train / Validation / Test) pour éviter la fuite d'information temporelle. (Assigné à : Membre 2)


# Phase 3 : Construction de la Topologie (Graphe Hybride)
• Tâche H : Calculer la matrice de corrélation (ex: corrélation de Pearson) entre les profils de consommation des 370 nœuds sur l'ensemble d'entraînement. (Assigné à : Membre 3)
• Tâche I : Définir un seuil de corrélation pertinent pour filtrer le bruit et générer la matrice d'adjacence finale (qui relie qui). (Assigné à : Membre 3)
• Tâche J : Visualiser le graphe généré (avec des outils comme NetworkX) pour s'assurer que la topologie a du sens avant de l'envoyer au réseau de neurones. (Assigné à : Membre 3)


# Phase 4 : Machine Learning Classique (Baseline Tabulaire)
• Tâche K : Entraîner la baseline avec l'algorithme XGBoost sur les données purement tabulaires d'entraînement. (Assigné à : Membre 2)
• Tâche L : Optimiser les hyperparamètres de XGBoost (profondeur des arbres, learning rate) sur l'ensemble de validation. (Assigné à : Membre 2)
• Tâche M : Évaluer le modèle XGBoost sur l'ensemble de Test et sauvegarder les prédictions brutes. (Assigné à : Membre 2)


# Phase 5 : Ingénierie Deep Learning et GNN
• Tâche N : Transformer les séries temporelles et la matrice d'adjacence en objets tenseurs Data compatibles avec la bibliothèque PyTorch Geometric. (Assigné à : Youssef)
• Tâche O : Coder l'architecture de la classe GCN en PyTorch (couches de convolution géométrique, fonctions d'activation, et couche de sortie pour la classification). (Assigné à : Youssef)
• Tâche P : Écrire la boucle d'entraînement (Training Loop), incluant la fonction de perte (Loss) et l'optimiseur (ex: Adam). (Assigné à : Youssef)


# Phase 6 : Entraînement du Graphe et Inférence
• Tâche Q : Lancer l'entraînement du modèle GCN et monitorer les courbes d'apprentissage pour éviter l'overfitting. (Assigné à : Youssef)
• Tâche R : Ajuster l'architecture si nécessaire (nombre de couches GCN, taille des vecteurs cachés). (Assigné à : Youssef)
• Tâche S : Générer et sauvegarder les prédictions finales du modèle GCN sur l'ensemble de Test. (Assigné à : Youssef)


# Phase 7 : Évaluation Comparative et Étude d'Ablation
• Tâche T : Calculer toutes les métriques de performance finales (Accuracy, Precision, Recall, F1-Score) pour XGBoost ET le modèle GCN. (Assigné à : Membre 3)
• Tâche U : Créer les visualisations comparatives (matrices de confusion, graphiques en barres des performances). (Assigné à : Membre 3)
• Tâche V : Rédiger la conclusion de l'étude d'ablation : quantifier mathématiquement ce que la structure en graphe a apporté de plus par rapport à XGBoost. (Assigné à : Membre 3 & Youssef)


# Phase 8 : Finalisation et Livrables
• Tâche W : Nettoyer le code Python, ajouter les commentaires finaux, et s'assurer que le script tourne de bout en bout sans erreur. (Tous les membres)
• Tâche X : Rédiger les sections Méthodologie, Résultats, et Discussion dans le document LaTeX. (Tous les membres, mené par Membre 3)
• Tâche Y : Extraire les éléments clés du rapport LaTeX pour concevoir les slides PowerPoint/Beamer pour la soutenance finale. (Tous les membres)
• Tâche Z : Relecture finale globale et soumission officielle des livrables au professeur. (Assigné à : Youssef)