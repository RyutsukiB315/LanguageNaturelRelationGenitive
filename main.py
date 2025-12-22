import os
import time
import warnings
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset
from Levenshtein import distance as lev_distance
import joblib
import torch

# --- NOUVEAU : Import de XGBoost pour le GPU ---
from xgboost import XGBClassifier

# On supprime les warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. GESTION DES DONNÉES
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
            return None

        files = [f for f in os.listdir(self.corpus_path) if f.endswith(".txt")]
        if not files:
            print("⚠️ Aucun fichier .txt trouvé.")
            return None

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
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("[") or line.lower().startswith("source") or line.startswith("A;B"):
                        continue

                    parts = line.split(';')
                    if len(parts) >= 2:
                        ta, tb = parts[0].strip(), parts[1].strip()
                        phrase = parts[2].strip() if len(parts) > 2 and len(parts[2].strip()) > 5 else f"{ta} de {tb}"

                        if (ta, tb, phrase) not in seen:
                            data_pairs.append((ta, tb))
                            data_context.append(phrase)
                            labels.append(label_id)
                            seen.add((ta, tb, phrase))

        print(f"✅ Données chargées : {len(data_pairs)} paires.")
        return {"pairs": data_pairs, "context": data_context, "labels": labels}


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
nlp = None


def init_spacy():
    global nlp
    if nlp is None:
        # Note: Spacy sur GPU pour des vecteurs simples 'sm' est souvent plus lent
        # à cause du transfert de données. On garde le CPU pour l'extraction,
        # mais on mettra l'entraînement (le plus lourd) sur GPU.
        try:
            nlp = spacy.load("fr_core_news_sm", disable=['parser', 'ner'])
        except OSError:
            print("⚠️ Téléchargement de Spacy...")
            os.system("python -m spacy download fr_core_news_sm")
            nlp = spacy.load("fr_core_news_sm", disable=['parser', 'ner'])


def extract_single_feature(args):
    term_a, term_b = args
    global nlp
    if nlp is None: init_spacy()

    doc_a = nlp(term_a)
    doc_b = nlp(term_b)

    vec_a = doc_a.vector if doc_a.has_vector else np.zeros(96)
    vec_b = doc_b.vector if doc_b.has_vector else np.zeros(96)

    sim = 0.0
    if doc_a.has_vector and doc_b.has_vector and doc_a.vector_norm > 0:
        sim = np.dot(vec_a, vec_b) / (doc_a.vector_norm * doc_b.vector_norm)

    lev = lev_distance(term_a, term_b)
    len_diff = len(term_a) - len(term_b)

    w_a = term_a.lower()
    suffixes = [
        1 if w_a.endswith("eur") else 0,
        1 if w_a.endswith("tion") else 0,
        1 if w_a.endswith("ment") else 0
    ]

    return np.concatenate([vec_a, vec_b, [sim, lev, len_diff], suffixes])


class FastFeatureExtractor:
    def __init__(self):
        init_spacy()

    def get_features_batch(self, data_pairs):
        # Parallélisation CPU pour l'extraction (plus stable que GPU pour joblib)
        return np.array(joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(extract_single_feature)(p) for p in data_pairs
        ))


# ==============================================================================
# 3. MODÈLE HYBRIDE (XGBoost sur GPU 🚀)
# ==============================================================================
class HybridModel:
    def __init__(self):
        self.extractor = FastFeatureExtractor()

        # DÉTECTION DU GPU POUR XGBOOST
        # Si CUDA est dispo, on configure XGBoost pour l'utiliser
        if torch.cuda.is_available():
            print("   🚀 [Hybrid] Configuration GPU (NVIDIA CUDA) activée.")
            device_type = "cuda"
            tree_method = "hist"  # Plus rapide sur GPU
        else:
            print("   🐢 [Hybrid] Pas de GPU détecté, repli sur CPU.")
            device_type = "cpu"
            tree_method = "auto"

        # Remplacement de GradientBoostingClassifier par XGBClassifier
        self.classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            # Paramètres GPU
            device=device_type,
            tree_method=tree_method,
            verbosity=0
        )

    def train(self, X_pairs, y):
        print(f"[Hybrid] Calcul des features...")
        X_vec = self.extractor.get_features_batch(X_pairs)
        print(f"[Hybrid] Entraînement XGBoost (Mode: {self.classifier.device})...")
        self.classifier.fit(X_vec, y)

    def evaluate(self, X_pairs, y):
        X_vec = self.extractor.get_features_batch(X_pairs)
        preds = self.classifier.predict(X_vec)
        return accuracy_score(y, preds)


# ==============================================================================
# 4. MODÈLE BERT (GPU 🚀)
# ==============================================================================
class BertModel:
    def __init__(self):
        # MPNet pour la qualité
        self.model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.model = None

    def train(self, X_text, y, X_test_text, y_test):
        print(f"\n[BERT] ⚙️ Configuration GPU...")

        if torch.cuda.is_available():
            device = "cuda"
            print("   🚀 [BERT] GPU (CUDA) détecté et activé.")
        else:
            device = "cpu"
            print("   🐢 [BERT] CPU uniquement (Attention, ce sera lent).")

        self.model = SetFitModel.from_pretrained(self.model_name)
        # Force le déplacement sur GPU
        self.model.to(device)

        train_ds = Dataset.from_dict({"text": X_text, "label": y})
        test_ds = Dataset.from_dict({"text": X_test_text, "label": y_test})

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            # Optimisation
            batch_size=16,
            num_iterations=10,
            num_epochs=1,
            column_mapping={"text": "text", "label": "label"}
        )

        print("[BERT] Démarrage de l'entraînement...")
        trainer.train()

        metrics = trainer.evaluate()
        return metrics.get('accuracy', metrics.get('test_accuracy', 0.0))


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Petit check initial
    if torch.cuda.is_available():
        print(f"✅ GPU DÉTECTÉ : {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ AUCUN GPU NVIDIA DÉTECTÉ. Assurez-vous d'avoir installé 'torch' avec support CUDA.")

    corpus_folder = os.path.join(os.getcwd(), "corpus15K")
    loader = CorpusLoader(corpus_folder)
    full_data = loader.load_data()

    if full_data and len(full_data["pairs"]) > 20:
        data_zipped = list(zip(full_data["pairs"], full_data["context"], full_data["labels"]))
        train_set, test_set = train_test_split(
            data_zipped, test_size=0.20, random_state=42, stratify=full_data["labels"]
        )

        X_train_pairs = [x[0] for x in train_set]
        X_train_ctx = [x[1] for x in train_set]
        y_train = [x[2] for x in train_set]

        X_test_pairs = [x[0] for x in test_set]
        X_test_ctx = [x[1] for x in test_set]
        y_test = [x[2] for x in test_set]

        print("\n" + "=" * 50)
        print("🚀 DÉMARRAGE ENTRAÎNEMENT (MODE FULL GPU)")
        print("=" * 50)

        # 1. HYBRIDE (XGBoost GPU)
        start_h = time.time()
        hybrid_ai = HybridModel()
        hybrid_ai.train(X_train_pairs, y_train)
        acc_h = hybrid_ai.evaluate(X_test_pairs, y_test)
        time_h = time.time() - start_h
        print(f"\n📊 Hybride (XGBoost): {acc_h:.2%} (Temps: {time_h:.2f}s)")

        # 2. BERT (SetFit GPU)
        start_b = time.time()
        bert_ai = BertModel()
        acc_b = bert_ai.train(X_train_ctx, y_train, X_test_ctx, y_test)
        time_b = time.time() - start_b
        print(f"\n📊 BERT: {acc_b:.2%} (Temps: {time_b:.2f}s)")

        # Sauvegarde
        print("\n" + "=" * 50)
        if acc_h >= acc_b:
            print(f"🏆 Victoire Hybride. Sauvegarde...")
            joblib.dump(hybrid_ai.classifier, "best_model.pkl")
        else:
            print(f"🏆 Victoire BERT. Sauvegarde...")
            bert_ai.model.save_pretrained("best_model_bert")

        joblib.dump(loader.id2label, "labels.pkl")
        print("✅ Terminé.")

    else:
        print("❌ Données insuffisantes.")