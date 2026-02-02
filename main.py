import os
import time
import warnings
import joblib
import torch
import winsound
import nlpaug.augmenter.word as naw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset

warnings.filterwarnings("ignore")


class DataAugmenter:
    def __init__(self):
        print("---  Initialisation du Générateur de Données (Augmentation) ---")
        self.aug = naw.ContextualWordEmbsAug(
            model_path='distilbert-base-multilingual-cased',
            action="substitute",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def augment_dataset(self, X_text, y_labels, n_aug=2):

        print(f"    Génération de {n_aug} variantes par phrase... ")
        augmented_text = []
        augmented_labels = []
        augmented_text.extend(X_text)
        augmented_labels.extend(y_labels)

        for text, label in zip(X_text, y_labels):
            try:
                variations = self.aug.augment(text, n=n_aug)
                # Gestion du format de retour (parfois str, parfois list)
                if isinstance(variations, str):
                    variations = [variations]

                for var in variations:
                    if var != text:
                        augmented_text.append(var)
                        augmented_labels.append(label)
            except:
                pass

        print(f"    Données augmentées : {len(X_text)} -> {len(augmented_text)} exemples.")
        return augmented_text, augmented_labels


class CorpusLoader:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.label_map = {}
        self.id2label = {}

    def load_data(self):
        print(f"---  Chargement depuis '{self.corpus_path}' ---")
        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path, exist_ok=True)
            return None

        files = [f for f in os.listdir(self.corpus_path) if f.endswith(".txt")]
        if not files:
            print(" Aucun fichier .txt trouvé.")
            return None

        label_names = sorted([f.replace(".txt", "") for f in files])
        self.label_map = {name: i for i, name in enumerate(label_names)}
        self.id2label = {i: name for name, i in self.label_map.items()}

        data_context = []
        labels = []
        seen_texts = set()

        for filename in files:
            label_id = self.label_map[filename.replace(".txt", "")]
            filepath = os.path.join(self.corpus_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("[") or line.lower().startswith("source"):
                        continue
                    parts = line.split(';')
                    if len(parts) >= 2:
                        phrase = parts[2].strip() if len(parts) > 2 and len(
                            parts[2].strip()) > 5 else f"{parts[0]} de {parts[1]}"
                        phrase = phrase.lower().strip()
                        if phrase not in seen_texts:
                            data_context.append(phrase)
                            labels.append(label_id)
                            seen_texts.add(phrase)

        print(f" Données chargées (Uniques) : {len(data_context)} exemples.")
        return {"context": data_context, "labels": labels}


class BertModel:
    def __init__(self):
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model = None

    def train(self, X_text, y, X_test_text, y_test, id2label):
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
            num_iterations=10,
            num_epochs=1,
            learning_rate=2e-5,
            column_mapping={"text": "text", "label": "label"}
        )

        print(f"[BERT] Entraînement sur {len(X_text)} phrases...")
        trainer.train()


        metrics_test = trainer.evaluate(test_ds)
        metrics_train = trainer.evaluate(train_ds)
        test_acc = metrics_test.get('accuracy', 0.0)
        train_acc = metrics_train.get('accuracy', 0.0)

        print("[VISUALISATION] Génération de la matrice de confusion...")
        y_pred = self.model.predict(X_test_text)

        cm = confusion_matrix(y_test, y_pred)

        labels_names = [id2label[i] for i in range(len(id2label))]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_names, yticklabels=labels_names)
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité Terrain')
        plt.title('Matrice de Confusion')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()  # Important pour libérer la mémoire
        print("   -> Sauvegardé : confusion_matrix.png")

        return train_acc, test_acc, train_acc - test_acc


if __name__ == "__main__":
    try:
        # Petit bip de démarrage
        winsound.Beep(440, 200)
    except:
        pass

    corpus_folder = os.path.join(os.getcwd(), "corpus80")
    loader = CorpusLoader(corpus_folder)
    full_data = loader.load_data()

    if full_data and len(full_data["context"]) > 20:
        X_train, X_test, y_train, y_test = train_test_split(
            full_data["context"],
            full_data["labels"],
            test_size=0.20,
            random_state=42,
            stratify=full_data["labels"]
        )

        # 2. Augmentation des données (Seulement sur le Train)
        augmenter = DataAugmenter()
        X_train_aug, y_train_aug = augmenter.augment_dataset(X_train, y_train, n_aug=2)

        print("\n" + "=" * 50)
        print(f"DÉMARRAGE (Train Augmenté: {len(X_train_aug)} | Test: {len(X_test)})")
        print("=" * 50)

        start_b = time.time()
        bert_ai = BertModel()

        train_acc, test_acc, gap = bert_ai.train(
            X_train_aug, y_train_aug, X_test, y_test, loader.id2label
        )

        time_b = time.time() - start_b

        print("\n" + "-" * 40)
        print(f" RÉSULTATS (Avec Data Augmentation)")
        print("-" * 40)
        print(f"  Temps total         : {time_b:.2f} s")
        print(f"  Train Accuracy      : {train_acc:.2%}")
        print(f"  Test Accuracy       : {test_acc:.2%}")
        print("-" * 40)

        print(f"\n Sauvegarde...")
        bert_ai.model.save_pretrained("best_model_augmented")
        joblib.dump(loader.id2label, "labels.pkl")
        print(" Terminé. (L'image 'confusion_matrix.png' a été créée)")

        try:
            winsound.Beep(523, 150)
            winsound.Beep(784, 300)
        except:
            pass
    else:
        print(" Pas assez de données.")