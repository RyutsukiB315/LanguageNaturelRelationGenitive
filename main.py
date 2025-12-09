import os
import numpy as np
import spacy
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Chargement du modèle spaCy
try:
    print("Chargement du modèle spaCy (fr_core_news_lg)...")
    nlp = spacy.load("fr_core_news_lg")
except OSError:
    print("Modèle introuvable. Téléchargement...")
    from spacy.cli import download

    download("fr_core_news_lg")
    nlp = spacy.load("fr_core_news_lg")


class GraspItVectorized:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.centroids = {}

    def get_vector(self, text):
        """Récupère le vecteur via spaCy"""
        return nlp(text).vector

    def cosine_similarity(self, vec_a, vec_b):
        """Calcul similarité cosinus"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def load_dataset(self):
        """
        1. Lit tous les fichiers
        2. Retourne une liste unique contenant toutes les données
        Format: [(term_a, term_b, vraie_relation), ...]
        """
        dataset = []
        print("\n--- Chargement du corpus ---")

        if not os.path.exists(self.corpus_path):
            print(f"ERREUR: Dossier {self.corpus_path} introuvable.")
            return []

        for filename in os.listdir(self.corpus_path):
            if filename.endswith(".txt"):
                relation_type = filename.replace(".txt", "")
                filepath = os.path.join(self.corpus_path, filename)

                count = 0
                with open(filepath, 'r', encoding='utf-8') as f:
                    next(f)  # Sauter l'en-tête
                    for line in f:
                        parts = line.strip().split(';')
                        if len(parts) >= 2:
                            term_a = parts[0].strip()
                            term_b = parts[1].strip()
                            # On stocke la donnée brute + son étiquette (la relation)
                            dataset.append((term_a, term_b, relation_type))
                            count += 1
                print(f"  -> '{relation_type}': {count} exemples chargés.")

        return dataset

    def train(self, training_data):
        """
        Phase d'apprentissage sur les 75% du corpus.
        Calcule les vecteurs moyens (centroïdes) pour chaque relation.
        """
        print(f"\n--- Entraînement sur {len(training_data)} exemples ---")

        # Stockage temporaire pour calculer les moyennes
        temp_data = defaultdict(lambda: {"A": [], "B": []})

        for term_a, term_b, relation in training_data:
            temp_data[relation]["A"].append(self.get_vector(term_a))
            temp_data[relation]["B"].append(self.get_vector(term_b))

        # Calcul des centröides finaux
        self.centroids = {}  # Reset
        for relation, vectors in temp_data.items():
            if vectors["A"] and vectors["B"]:
                mean_a = np.mean(vectors["A"], axis=0)
                mean_b = np.mean(vectors["B"], axis=0)
                self.centroids[relation] = {"A": mean_a, "B": mean_b}

        print("--- Entraînement terminé ---")

    def predict(self, term_a, term_b):
        """Prédiction standard"""
        vec_a = self.get_vector(term_a)
        vec_b = self.get_vector(term_b)

        best_score = -1
        best_relation = "Inconnu"
        details = []

        for relation, means in self.centroids.items():
            sim_a = self.cosine_similarity(vec_a, means["A"])
            sim_b = self.cosine_similarity(vec_b, means["B"])
            avg_score = (sim_a + sim_b) / 2

            details.append((relation, avg_score))

            if avg_score > best_score:
                best_score = avg_score
                best_relation = relation

        details.sort(key=lambda x: x[1], reverse=True)
        return best_relation, best_score, details[:3]

    def evaluate(self, test_data):
        """
        Phase de validation sur les 25% restants.
        Vérifie si la prédiction correspond à la vraie étiquette.
        """
        print(f"\n--- Évaluation du modèle sur {len(test_data)} exemples de test ---")

        correct = 0
        total = len(test_data)
        errors = []

        for term_a, term_b, true_label in test_data:
            predicted_label, score, _ = self.predict(term_a, term_b)

            if predicted_label == true_label:
                correct += 1
            else:
                # On note l'erreur pour analyse
                errors.append(f"{term_a}-{term_b}: Prédit '{predicted_label}' au lieu de '{true_label}'")

        accuracy = (correct / total) * 100
        print(f"Résultat : {correct}/{total} corrects")
        print(f"PRÉCISION GLOBALE (Accuracy) : {accuracy:.2f}%")

        # Afficher quelques erreurs pour comprendre
        if errors:
            print("\nExemples d'erreurs :")
            for err in errors[:5]:  # Afficher les 5 premières erreurs
                print(f" - {err}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    corpus_folder = os.path.join(os.getcwd(), "corpus")
    ai = GraspItVectorized(corpus_folder)

    # 1. Charger tout le dataset
    full_dataset = ai.load_dataset()

    if full_dataset:
        # 2. Séparer en Train (75%) et Test (25%)
        # stratify=None : mélange aléatoire pur
        # random_state=42 : pour avoir toujours le même mélange (reproductible)
        train_set, test_set = train_test_split(full_dataset, test_size=0.25, random_state=42)

        # 3. Entraîner UNIQUEMENT sur le train_set
        ai.train(train_set)

        # 4. Vérifier le modèle sur le test_set (données jamais vues)
        ai.evaluate(test_set)

        # 5. Test manuel (optionnel, pour le fun)
        print("\n--- Test Manuel Rapide ---")
        rel, score, _ = ai.predict("tristesse", "visage")
        print(f"Test 'tristesse de visage' -> {rel} ({score:.2f})")