import os
import joblib
import torch
from setfit import SetFitModel
import warnings

# Ignorer les warnings inutiles
warnings.filterwarnings("ignore")


def load_inference_system():
    print("--- ⚙️ Chargement du modèle et des labels ---")

    model_path = "best_model_bert"
    label_path = "labels.pkl"

    # 1. Vérification des fichiers
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le dossier modèle '{model_path}' est introuvable.")
        return None, None
    if not os.path.exists(label_path):
        print(f"❌ Erreur : Le fichier labels '{label_path}' est introuvable.")
        return None, None

    # 2. Chargement du modèle SetFit
    # Utilisation du GPU si dispo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔌 Device utilisé : {device}")

    try:
        model = SetFitModel.from_pretrained(model_path)
        model.to(device)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None

    # 3. Chargement des labels
    try:
        id2label = joblib.load(label_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement des labels : {e}")
        return None, None

    print("✅ Système prêt.\n")
    return model, id2label


def predict_loop(model, id2label):
    print("==================================================")
    print("🧠 MODE TEST INTERACTIF (Tapez 'exit' pour quitter)")
    print("==================================================")

    while True:
        user_input = input("\n📝 Entrez une phrase ou expression : ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Au revoir !")
            break

        if not user_input:
            continue

        # --- PRÉDICTION ---
        # predict renvoie la classe, predict_proba renvoie les probabilités
        # SetFit attend une liste, donc on met [user_input]
        preds = model.predict([user_input])
        probs = model.predict_proba([user_input])

        # Récupération de l'index prédit (c'est un tenseur ou un entier)
        pred_idx = int(preds[0])

        # Récupération du nom du label
        label_name = id2label.get(pred_idx, "Inconnu")

        # Récupération de la confiance (score)
        confidence = probs[0][pred_idx].item()

        # --- AFFICHAGE ---
        print(f"   Label prédit : \033[1m{label_name}\033[0m")
        print(f"   Confiance    : {confidence:.2%}")


if __name__ == "__main__":
    # 1. Charger
    ai_model, labels_map = load_inference_system()

    # 2. Lancer la boucle si tout est OK
    if ai_model and labels_map:
        predict_loop(ai_model, labels_map)