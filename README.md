
   CLASSIFICATION DE RELATIONS SEMANTIQUES (GENITIVE) AVEC SETFIT & NLP AUG
==============================================================================

Ce projet permet d'entrainer une IA pour classifier des relations sémantiques 
dans des phrases. Il utilise l'augmentation de données (synonymes) pour 
améliorer les résultats.

------------------------------------------------------------------------------
1. PREREQUIS MATERIEL (IMPORTANT)
------------------------------------------------------------------------------
Ce code est conçu pour fonctionner sur une machine équipée d'une :
CARTE GRAPHIQUE NVIDIA (GPU) compatible CUDA.

Pourquoi ?
L'étape d'expansion des données (génération de milliers de synonymes) et 
l'entraînement du modèle BERT demandent beaucoup de puissance de calcul. 
Sur un processeur classique (CPU), l'exécution sera extrêmement lente.

------------------------------------------------------------------------------
2. INSTALLATION
------------------------------------------------------------------------------
Ouvrez votre terminal et installez les librairies Python nécessaires :

pip install torch setfit sentence-transformers scikit-learn nlpaug matplotlib seaborn datasets joblib

------------------------------------------------------------------------------
3. UTILISATION (ORDRE D'EXECUTION)
------------------------------------------------------------------------------
Pour que le projet fonctionne, respectez cet ordre précis :

ETAPE 1 : PREPARATION
Commande : ``` python extract_and_save_connectors.py```
Action   : Extrait les connecteurs ou prépare les données brutes nécessaires.

ETAPE 2 : ENTRAINEMENT
Commande : ```python main.py```
Action   : Charge le corpus, crée des synonymes, entraîne l'IA et génère 
           les graphiques.
Résultat : Crée le dossier "best_model_augmented", le fichier "labels.pkl" 
           et l'image "confusion_matrix.png".

ETAPE 3 : TEST & INFERENCE
Commande : ```python test.py```
Action   : Charge le modèle que vous venez de créer pour tester de 
           nouvelles phrases.

------------------------------------------------------------------------------
4. FONCTIONNEMENT DETAILLE DU CODE (MAIN.PY)
------------------------------------------------------------------------------
Le script "main.py" suit une méthodologie rigoureuse étape par étape :

1. Initialisation : Chargement des librairies et du modèle de langage 
   (DistilBERT) qui servira à inventer des variantes de phrases.

2. Ingestion      : Lecture automatique des fichiers textes situés dans le 
   dossier "corpus80". Le nom du fichier définit la catégorie.

3. Nettoyage      : Suppression automatique des doublons et filtration des 
   lignes erronées pour ne garder que des données propres.

4. Séparation     : Division des données : 80% pour l'entraînement de l'IA 
   et 20% mis de côté pour l'examen final (Test).

5. Expansion      : Utilisation de NLPAug pour enrichir le vocabulaire. 
   Le script génère des variantes des phrases d'entraînement en remplaçant 
   des mots par des synonymes.

6. Configuration  : Chargement du modèle BERT (MiniLM) et réglage des 
   paramètres d'apprentissage.

7. Apprentissage  : Lancement de l'entraînement (Fine-tuning) sur les 
   données augmentées.

8. Evaluation     : Mesure de la performance sur les 20% de données cachées 
   et création de la Matrice de Confusion (image) pour voir les erreurs.

9. Exportation    : Sauvegarde du cerveau de l'IA (le modèle) et du fichier 
   de configuration des labels pour pouvoir l'utiliser plus tard.

------------------------------------------------------------------------------
5. STRUCTURE DES DOSSIERS
------------------------------------------------------------------------------
```text
/Projet
  ├── corpus80/                       # (Données .txt sources)
  ├── extract_and_save_connectors.py  # (Script Etape 1 : Préparation)
  ├── main.py                         # (Script Etape 2 : Entraînement)
  ├── test.py                         # (Script Etape 3 : Test)
  ├── best_model_augmented/           # [GÉNÉRÉ] Dossier du modèle après entraînement
  ├── labels.pkl                      # [GÉNÉRÉ] Fichier de configuration des labels
  └── confusion_matrix.png            # [GÉNÉRÉ] Image du diagnostic (Matrice)
