import os
import time
import warnings
import joblib
import torch
import winsound  # Pour les sons sur Windows

from sklearn.model_selection import train_test_split
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset

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

        # On ne garde que le contexte (la phrase) et le label pour BERT
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
                        # La phrase est ce qui intéresse BERT
                        phrase = parts[2].strip() if len(parts) > 2 and len(parts[2].strip()) > 5 else f"{ta} de {tb}"

                        if (phrase, label_id) not in seen:
                            data_context.append(phrase)
                            labels.append(label_id)
                            seen.add((phrase, label_id))

        print(f"✅ Données chargées : {len(data_context)} exemples.")
        return {"context": data_context, "labels": labels}


# ==============================================================================
# 2. MODÈLE BERT (SetFit)
# ==============================================================================
class BertModel:
    def __init__(self):
        self.model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.model = None

    def train(self, X_text, y, X_test_text, y_test):
        print(f"\n[BERT] ⚙️ Configuration GPU...")

        if torch.cuda.is_available():
            device = "cuda"
            print("   🚀 [BERT] GPU (CUDA) détecté et activé.")
        else:
            device = "cpu"
            print("   🐢 [BERT] CPU uniquement.")

        self.model = SetFitModel.from_pretrained(self.model_name)
        self.model.to(device)

        train_ds = Dataset.from_dict({"text": X_text, "label": y})
        test_ds = Dataset.from_dict({"text": X_test_text, "label": y_test})

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=16,
            num_iterations=20,  # Génération de paires positives/négatives
            num_epochs=1,  # 1 époque suffit pour la tête de classification
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
    # ==========================
    # 🔊 BIP DE DÉMARRAGE
    # ==========================
    try:
        print("🔊 Initialisation...")
        winsound.Beep(440, 300)
    except:
        pass

    if torch.cuda.is_available():
        print(f"✅ GPU DÉTECTÉ : {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ AUCUN GPU NVIDIA DÉTECTÉ.")

    corpus_folder = os.path.join(os.getcwd(), "corpus80")
    loader = CorpusLoader(corpus_folder)
    full_data = loader.load_data()

    if full_data and len(full_data["context"]) > 20:
        # Préparation des données uniquement texte
        data_zipped = list(zip(full_data["context"], full_data["labels"]))

        train_set, test_set = train_test_split(
            data_zipped, test_size=0.20, random_state=42, stratify=full_data["labels"]
        )

        # Extraction propre
        X_train_ctx = [x[0] for x in train_set]
        y_train = [x[1] for x in train_set]

        X_test_ctx = [x[0] for x in test_set]
        y_test = [x[1] for x in test_set]

        print("\n" + "=" * 50)
        print("🚀 DÉMARRAGE ENTRAÎNEMENT BERT (SETFIT)")
        print("=" * 50)

        start_b = time.time()
        bert_ai = BertModel()
        acc_b = bert_ai.train(X_train_ctx, y_train, X_test_ctx, y_test)
        time_b = time.time() - start_b

        print(f"\n📊 Résultat BERT: {acc_b:.2%} (Temps: {time_b:.2f}s)")
        print(f"💾 Sauvegarde du modèle...")

        # Sauvegarde du modèle et des labels
        bert_ai.model.save_pretrained("best_model_bert")
        joblib.dump(loader.id2label, "labels.pkl")

        print("✅ Terminé.")

        # ==========================
        # 🔊 SON DE FIN
        # ==========================
        try:
            print("\n🔊 Fin du traitement !")
            winsound.Beep(523, 150)  # Do
            time.sleep(0.05)
            winsound.Beep(659, 150)  # Mi
            time.sleep(0.05)
            winsound.Beep(784, 300)  # Sol




        except Exception:
            pass
    else:
        print("❌ Données insuffisantes.")
        try:
            winsound.Beep(200, 500)
        except:
            pass