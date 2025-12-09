import os
import numpy as np
import spacy
from collections import defaultdict

# Chargement du modèle de langue français avec vecteurs (Word2Vec/GloVe intégrés)
# Assurez-vous d'avoir lancé : python -m spacy download fr_core_news_lg
try:
    print("Chargement du modèle spaCy (fr_core_news_lg)...")
    nlp = spacy.load("fr_core_news_lg")
except OSError:
    print("Modèle 'fr_core_news_lg' introuvable. Téléchargement en cours...")
    from spacy.cli import download

    download("fr_core_news_lg")
    nlp = spacy.load("fr_core_news_lg")


class GraspItVectorized:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        # Stockage des "Centröides" (Vecteurs moyens) pour chaque relation
        # Structure : { "Relation": { "A": vector_moyen_A, "B": vector_moyen_B } }
        self.centroids = {}

    def get_vector(self, text):
        """Récupère le vecteur (embedding) d'un mot via spaCy"""
        # On traite le texte et on récupère le vecteur du document entier
        # (moyenne des mots si composé, ex: "pomme de terre")
        return nlp(text).vector

    def cosine_similarity(self, vec_a, vec_b):
        """Calcul optimisé de la similarité cosinus avec NumPy"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def train(self):
        """
        Phase d'apprentissage :
        Au lieu de fusionner des règles symboliques, on calcule le vecteur MOYEN
        de tous les termes A et de tous les termes B pour chaque relation.
        """
        print("\n--- Démarrage de l'entraînement vectoriel ---")

        # Dictionnaire temporaire pour stocker toutes les listes de vecteurs
        temp_data = defaultdict(lambda: {"A": [], "B": []})

        # Lecture des fichiers
        for filename in os.listdir(self.corpus_path):
            if filename.endswith(".txt"):
                relation_type = filename.replace(".txt", "")
                filepath = os.path.join(self.corpus_path, filename)

                count = 0
                with open(filepath, 'r', encoding='utf-8') as f:
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split(';')
                        if len(parts) >= 2:
                            term_a = parts[0].strip()
                            term_b = parts[1].strip()

                            # On ajoute les vecteurs à la liste
                            temp_data[relation_type]["A"].append(self.get_vector(term_a))
                            temp_data[relation_type]["B"].append(self.get_vector(term_b))
                            count += 1

                print(f"  -> Modèle '{relation_type}' entraîné sur {count} exemples.")

        # Calcul des centröides (Moyenne des vecteurs)
        for relation, vectors in temp_data.items():
            if vectors["A"] and vectors["B"]:
                # axis=0 permet de calculer la moyenne colonne par colonne (dimension par dimension)
                mean_a = np.mean(vectors["A"], axis=0)
                mean_b = np.mean(vectors["B"], axis=0)

                self.centroids[relation] = {
                    "A": mean_a,
                    "B": mean_b
                }
        print("--- Entraînement terminé ---\n")

    def predict(self, term_a, term_b):
        """
        Phase de Prédiction :
        Compare les vecteurs d'entrée aux centröides appris.
        """
        vec_a = self.get_vector(term_a)
        vec_b = self.get_vector(term_b)

        best_score = -1
        best_relation = "Inconnu"
        details = []

        for relation, means in self.centroids.items():
            # Similarité du terme A avec le prototype A de la relation
            sim_a = self.cosine_similarity(vec_a, means["A"])
            # Similarité du terme B avec le prototype B de la relation
            sim_b = self.cosine_similarity(vec_b, means["B"])

            # Score global (moyenne des deux similarités)
            avg_score = (sim_a + sim_b) / 2

            details.append((relation, avg_score))

            if avg_score > best_score:
                best_score = avg_score
                best_relation = relation

        # Tri des résultats pour affichage (optionnel)
        details.sort(key=lambda x: x[1], reverse=True)

        return best_relation, best_score, details[:3]  # Retourne le top 3


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Chemin vers le dossier corpus
    corpus_folder = os.path.join(os.getcwd(), "corpus")

    # Instanciation
    ai = GraspItVectorized(corpus_folder)

    # 1. Entraînement
    try:
        ai.train()

        # 2. Tests de prédiction
        print("--- Tests de prédiction ---")
        tests = [
            ("cuillère", "bois"),  # Matiere
            ("couteau", "acier"),  # Matiere (acier n'est peut-être pas dans le corpus, mais proche vectoriellement)
            ("tristesse", "visage"),  # Caracterisation / Consequence
            ("aboiement", "chien"),  # Agent
            ("clé", "porte"),  # Instrument
            ("appartement", "Paris"),  # Lieu
            ("film", "horreur")  # Topic
        ]

        for t_a, t_b in tests:
            rel, score, top3 = ai.predict(t_a, t_b)
            print(f"'{t_a} de {t_b}'")
            print(f"   -> PRÉDICTION : {rel.upper()} (Confiance: {score:.2f})")
            print(f"   -> Top 3: {[(r, round(s, 2)) for r, s in top3]}")
            print("-" * 30)

    except FileNotFoundError:
        print(f"ERREUR : Le dossier '{corpus_folder}' est introuvable.")