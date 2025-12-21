import sys
import re
import joblib
import numpy as np
import spacy
from Levenshtein import distance as lev_distance
from setfit import SetFitModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ==============================================================================
# 1. FEATURE EXTRACTOR (Doit être IDENTIQUE à celui de l'entraînement)
# ==============================================================================
class FeatureExtractor:
    def __init__(self):
        # On charge le même modèle Spacy que lors de l'entraînement
        try:
            self.nlp = spacy.load("fr_core_news_lg", disable=['parser'])
        except OSError:
            self.nlp = spacy.load("fr_core_news_md", disable=['parser'])

        # Cache pour accélérer les tests répétés
        self.word_cache = {}

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
        # On utilise le cache si possible
        doc_a = self.word_cache.get(term_a, self.nlp(term_a))
        doc_b = self.word_cache.get(term_b, self.nlp(term_b))

        # Mise en cache pour la prochaine fois
        self.word_cache[term_a] = doc_a
        self.word_cache[term_b] = doc_b

        # 1. Vecteurs
        vec_a = doc_a.vector if doc_a.has_vector else np.zeros(300)
        vec_b = doc_b.vector if doc_b.has_vector else np.zeros(300)

        sim = 0.0
        if doc_a.has_vector and doc_b.has_vector and doc_a.vector_norm > 0:
            sim = doc_a.similarity(doc_b)

        # 2. Morphologie
        lev = lev_distance(term_a, term_b)
        len_diff = len(term_a) - len(term_b)

        # 3. Grammaire & NER
        pos_a = doc_a[0].pos if len(doc_a) > 0 else 0
        pos_b = doc_b[0].pos if len(doc_b) > 0 else 0
        is_propn = 1 if pos_b == 96 else 0

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
# 2. CERVEAU (Moteur d'inférence)
# ==============================================================================
class Brain:
    def __init__(self):
        console.print("[cyan]Chargement des modèles...[/]")
        try:
            self.id2label = joblib.load("labels.pkl")
            # On essaie de charger le meilleur modèle disponible
            try:
                self.clf = joblib.load("best_model.pkl")
                self.modele_actif = "Hybride"
            except:
                self.clf = None

            try:
                self.bert = SetFitModel.from_pretrained("best_model_bert")
                if self.clf is None: self.modele_actif = "BERT"
            except:
                self.bert = None

        except FileNotFoundError:
            console.print("[bold red]❌ Erreur : Fichiers manquants. Lancez d'abord l'entraînement ![/]")
            sys.exit(1)

        self.extractor = FeatureExtractor()
        console.print("[green]✅ Système prêt ![/]")

    def analyze(self, phrase):
        # Découpage basique de la phrase "A de B"
        match = re.search(r"^(.*?)\s+(?:de|du|d'|des|en)\s+(.*)$", phrase, re.IGNORECASE)

        if not match:
            console.print(f"[yellow]⚠️ Structure non reconnue : '{phrase}' (essayez 'X de Y')[/]")
            return

        ta, tb = match.group(1), match.group(2)

        # --- 1. HYBRIDE (Si disponible) ---
        res_h, prob_h = "N/A", 0.0
        if self.clf:
            feats = self.extractor.get_features(ta, tb)
            pred_id = self.clf.predict([feats])[0]
            # Probabilités
            probas = self.clf.predict_proba([feats])[0]
            prob_h = float(np.max(probas))
            res_h = self.id2label[int(pred_id)]

        # --- 2. BERT (Si disponible) ---
        res_b, prob_b = "N/A", 0.0
        if self.bert:
            # C'EST ICI LA CORRECTION : Conversion tensor -> numpy -> int
            probas_tensor = self.bert.predict_proba([phrase])[0]

            # On extrait les valeurs proprement
            if hasattr(probas_tensor, 'detach'):
                probas_numpy = probas_tensor.detach().cpu().numpy()
            else:
                probas_numpy = probas_tensor  # Déjà numpy si CPU

            pred_b_id = np.argmax(probas_numpy)
            prob_b = float(np.max(probas_numpy))

            # Conversion explicite en int pour éviter le KeyError: tensor(1)
            res_b = self.id2label[int(pred_b_id)]

        # --- AFFICHAGE ---
        table = Table(title=f"Analyse : [bold white]{phrase}[/]")
        table.add_column("Modèle", style="cyan")
        table.add_column("Résultat", style="magenta")
        table.add_column("Confiance", justify="right")

        if self.clf: table.add_row("🧠 Hybride", res_h, f"{prob_h:.1%}")
        if self.bert: table.add_row("🤖 BERT", res_b, f"{prob_b:.1%}")

        console.print(table)

        # Logique de décision simple (Consensus ou Meilleur score)
        final = res_h
        if self.bert and prob_b > prob_h:
            final = res_b

        color = "green" if res_h == res_b else "yellow"
        console.print(Panel(f"[bold {color}]DÉCISION : {final}[/]", expand=False))


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    ai = Brain()
    console.rule("[bold blue]MODE INTERACTIF[/]")
    console.print("Tapez 'q' pour quitter.\n")

    while True:
        txt = console.input("[bold]Phrase :[/] ")
        if txt.lower() in ['q', 'quit', 'exit']: break
        if len(txt.strip()) > 2:
            ai.analyze(txt)