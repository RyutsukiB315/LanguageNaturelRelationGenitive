import os
import time
import warnings
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset
from tqdm import tqdm
from Levenshtein import distance as lev_distance
import joblib

# On ignore les warnings pour garder la console propre
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. GESTION DES DONNÉES (Chargement & Nettoyage)
# ==============================================================================
class CorpusLoader:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.label_map = {}
        self.id2label = {}

    def load_data(self):
        print(f"--- 📂 Chargement depuis '{self.corpus_path}' ---")
        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path, exist_ok=True)
            print(f"⚠️ Dossier créé. Veuillez y placer vos fichiers .txt")
            return None

        files = [f for f in os.listdir(self.corpus_path) if f.endswith(".txt")]
        if not files:
            print("⚠️ Aucun fichier .txt trouvé.")
            return None

        # Création des labels
        label_names = sorted([f.replace(".txt", "") for f in files])
        self.label_map = {name: i for i, name in enumerate(label_names)}
        self.id2label = {i: name for name, i in self.label_map.items()}

        data_pairs = []
        data_context = []
        labels = []
        seen = set()

        for filename in files:
            label_id = self.label_map[filename.replace(".txt", "")]
            filepath = os.path.join(self.corpus_path, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    # Filtre : ignore lignes vides, sources, entêtes
                    if not line or line.startswith("[") or line.lower().startswith("source") or line.startswith("A;B"):
                        continue

                    parts = line.split(';')
                    if len(parts) >= 2:
                        ta, tb = parts[0].strip(), parts[1].strip()
                        # Récupération ou création de la phrase contextuelle
                        if len(parts) > 2 and len(parts[2].strip()) > 5:
                            phrase = parts[2].strip()
                        else:
                            phrase = f"{ta} de {tb}"

                        if (ta, tb) not in seen:
                            data_pairs.append((ta, tb))
                            data_context.append(phrase)
                            labels.append(label_id)
                            seen.add((ta, tb))

        print(f"✅ Données chargées : {len(data_pairs)} paires.")
        return {"pairs": data_pairs, "context": data_context, "labels": labels}


# ==============================================================================
# 2. FEATURE ENGINEERING OPTIMISÉ (Batch Processing)
# ==============================================================================
class FeatureExtractor:
    def __init__(self):
        print("[Features] Chargement de SpaCy...")
        try:
            # On désactive le 'parser' syntaxique qui est lent et inutile ici
            # Mais on garde 'ner' (entités) et 'tagger' (POS)
            self.nlp = spacy.load("fr_core_news_lg", disable=['parser'])
        except OSError:
            print("⚠️ Modèle 'lg' absent. Chargement de 'md'...")
            try:
                self.nlp = spacy.load("fr_core_news_md", disable=['parser'])
            except OSError:
                print("❌ ERREUR CRITIQUE : Aucun modèle Spacy trouvé.")
                raise

        self.word_cache = {}

    def prepare_batch(self, data_pairs):
        """
        OPTIMISATION VITESSE : Traite tous les mots d'un coup.
        """
        # Récupérer tous les mots uniques
        all_words = list(set([p[0] for p in data_pairs] + [p[1] for p in data_pairs]))

        # Si déjà en cache, on ne refait pas
        words_to_process = [w for w in all_words if w not in self.word_cache]

        if words_to_process:
            print(f"   ⚡ Traitement par lot de {len(words_to_process)} nouveaux mots...")
            # nlp.pipe est beaucoup plus rapide que nlp() en boucle
            docs = list(self.nlp.pipe(words_to_process, batch_size=2000))

            for w, doc in zip(words_to_process, docs):
                self.word_cache[w] = doc

    def get_suffix_features(self, word):
        w = word.lower()
        return [
            1 if w.endswith("eur") else 0,
            1 if w.endswith("té") else 0,
            1 if w.endswith("esse") else 0,
            1 if w.endswith("tion") else 0,
            1 if w.endswith("ment") else 0
        ]

    def get_features(self, term_a, term_b):
        # Récupération depuis le cache (ultra rapide)
        # Si mot inconnu (cas rare en test), on traite à la volée
        doc_a = self.word_cache.get(term_a, self.nlp(term_a))
        doc_b = self.word_cache.get(term_b, self.nlp(term_b))

        # 1. Vecteurs
        vec_a = doc_a.vector if doc_a.has_vector else np.zeros(300)
        vec_b = doc_b.vector if doc_b.has_vector else np.zeros(300)

        sim = 0.0
        if doc_a.has_vector and doc_b.has_vector and doc_a.vector_norm > 0:
            sim = doc_a.similarity(doc_b)

        # 2. Morphologie
        lev = lev_distance(term_a, term_b)
        len_diff = len(term_a) - len(term_b)

        # 3. Grammaire & NER (Optimisé)
        pos_a = doc_a[0].pos if len(doc_a) > 0 else 0
        pos_b = doc_b[0].pos if len(doc_b) > 0 else 0
        is_propn = 1 if pos_b == 96 else 0  # 96 = PROPN

        # Distinction Auteur vs Lieu via NER
        ent_per = 0
        ent_loc = 0
        if doc_b.ents:
            lbl = doc_b.ents[0].label_
            if lbl == "PER": ent_per = 1
            if lbl in ["LOC", "GPE"]: ent_loc = 1

        suff_a = self.get_suffix_features(term_a)

        return np.concatenate([
            vec_a, vec_b, (vec_a - vec_b),
            [sim, lev, len_diff, pos_a, pos_b, is_propn, ent_per, ent_loc],
            suff_a
        ])


# ==============================================================================
# 3. MODÈLE HYBRIDE (Gradient Boosting Optimisé)
# ==============================================================================
class HybridModel:
    def __init__(self):
        self.extractor = FeatureExtractor()
        # Paramètres allégés pour la vitesse
        self.classifier = GradientBoostingClassifier(
            n_estimators=150,  # Suffisant pour converger
            learning_rate=0.1,
            max_depth=4,  # Moins profond = plus rapide
            random_state=42,
            verbose=0
        )

    def prepare_data(self, data_pairs, is_training=False):
        # On lance le traitement par lot AVANT la boucle
        if is_training:
            self.extractor.prepare_batch(data_pairs)

        X = []
        # Utilisation du cache
        for ta, tb in data_pairs:
            feats = self.extractor.get_features(ta, tb)
            X.append(feats)
        return np.array(X)

    def train(self, X_pairs, y):
        print(f"[Hybrid] Calcul des features...")
        X_vec = self.prepare_data(X_pairs, is_training=True)
        print(f"[Hybrid] Entraînement Boosting ({X_vec.shape[0]} samples)...")
        self.classifier.fit(X_vec, y)

    def evaluate(self, X_pairs, y):
        # Pour le test, on s'assure que les mots sont cachés aussi
        self.extractor.prepare_batch(X_pairs)
        X_vec = self.prepare_data(X_pairs, is_training=False)
        preds = self.classifier.predict(X_vec)
        return accuracy_score(y, preds), preds

    def predict(self, ta, tb, id2label):
        # Prédiction unique
        feats = self.extractor.get_features(ta, tb)
        pred = self.classifier.predict([feats])[0]
        return id2label.get(pred, "Inconnu")


# ==============================================================================
# 4. MODÈLE DEEP LEARNING (SetFit / BERT Optimisé)
# ==============================================================================
class BertModel:
    def __init__(self):
        # MODÈLE RAPIDE (MiniLM) au lieu de MPNet
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model = None

    def train(self, X_text, y, X_test_text, y_test):
        print(f"\n[BERT] Chargement de {self.model_name}...")
        self.model = SetFitModel.from_pretrained(self.model_name)

        train_ds = Dataset.from_dict({"text": X_text, "label": y})
        test_ds = Dataset.from_dict({"text": X_test_text, "label": y_test})

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=16,
            num_iterations=20,  # RÉDUIT POUR LA VITESSE (20 suffisent souvent)
            column_mapping={"text": "text", "label": "label"}
        )

        print("[BERT] Fine-tuning en cours...")
        trainer.train()
        metrics = trainer.evaluate()
        return metrics.get('accuracy', metrics.get('test_accuracy', 0.0))

    def predict(self, text, id2label):
        pred = self.model.predict([text])[0]
        return id2label.get(int(pred), "Inconnu")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    corpus_folder = os.path.join(os.getcwd(), "corpus")
    loader = CorpusLoader(corpus_folder)
    full_data = loader.load_data()

    if full_data and len(full_data["pairs"]) > 20:
        # Split des données
        data_zipped = list(zip(full_data["pairs"], full_data["context"], full_data["labels"]))
        train_set, test_set = train_test_split(
            data_zipped, test_size=0.20, random_state=42, stratify=full_data["labels"]
        )

        # Extraction des listes
        X_train_pairs = [x[0] for x in train_set]
        X_train_ctx = [x[1] for x in train_set]
        y_train = [x[2] for x in train_set]

        X_test_pairs = [x[0] for x in test_set]
        X_test_ctx = [x[1] for x in test_set]
        y_test = [x[2] for x in test_set]

        print("\n" + "=" * 50)
        print("⏱️ DÉBUT DES ENTRAÎNEMENTS")
        print("=" * 50)

        # --- 1. HYBRIDE ---
        start_h = time.time()
        hybrid_ai = HybridModel()
        hybrid_ai.train(X_train_pairs, y_train)
        acc_h, preds_h = hybrid_ai.evaluate(X_test_pairs, y_test)
        time_h = time.time() - start_h

        print(f"\n📊 Hybride terminé en {time_h:.1f}s | Accuracy: {acc_h:.2%}")
        # print(classification_report(y_test, preds_h, target_names=list(loader.label_map.keys())))

        # --- 2. BERT ---
        start_b = time.time()
        bert_ai = BertModel()
        acc_b = bert_ai.train(X_train_ctx, y_train, X_test_ctx, y_test)
        time_b = time.time() - start_b

        print(f"\n📊 BERT terminé en {time_b:.1f}s | Accuracy: {acc_b:.2%}")

        # --- CONCLUSION ---
        print("\n" + "=" * 50)
        print(f"🏆 RÉSULTATS FINAUX")
        print(f"1. Hybride : {acc_h:.2%} (Temps: {time_h:.0f}s)")
        print(f"2. BERT    : {acc_b:.2%} (Temps: {time_b:.0f}s)")

        best_model = "Hybride" if acc_h > acc_b else "BERT"
        print(f"--> Meilleur modèle : {best_model}")

        # Sauvegarde
        if acc_h > acc_b:
            joblib.dump(hybrid_ai.classifier, "best_model.pkl")
            joblib.dump(loader.id2label, "labels.pkl")
            print("💾 Modèle Hybride sauvegardé.")
        else:
            bert_ai.model.save_pretrained("best_model_bert")
            joblib.dump(loader.id2label, "labels.pkl")
            print("💾 Modèle BERT sauvegardé.")

        # Test rapide
        ta, tb = "roman", "Zola"
        print(f"\nTest '{ta}' -> '{tb}':")
        print(f"Hybride: {hybrid_ai.predict(ta, tb, loader.id2label)}")
        print(f"BERT   : {bert_ai.predict(f'{ta} de {tb}', loader.id2label)}")

    else:
        print("❌ Pas assez de données.")