import os
from collections import Counter
import re


def scan_connectors(corpus_path):
    print(f"--- Analyse des connecteurs dans : {corpus_path} ---")

    if not os.path.exists(corpus_path):
        print("❌ Le dossier corpus15K n'existe pas.")
        return []

    files = [f for f in os.listdir(corpus_path) if f.endswith(".txt")]
    all_connectors = []

    for filename in files:
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            # On saute l'en-tête
            next(f, None)

            for line in f:
                parts = line.strip().split(';')

                # On a besoin de 3 colonnes : A; B; Contexte
                if len(parts) >= 3:
                    term_a = parts[0].strip()
                    term_b = parts[1].strip()
                    context = parts[2].strip()

                    # Vérification : le contexte contient-il A et B ?
                    lower_ctx = context.lower()
                    lower_a = term_a.lower()
                    lower_b = term_b.lower()

                    if lower_ctx.startswith(lower_a) and lower_ctx.endswith(lower_b):
                        # Découpage chirurgical
                        start_idx = len(term_a)
                        end_idx = len(context) - len(term_b)

                        raw_connector = context[start_idx:end_idx]
                        clean_connector = raw_connector.strip()

                        if clean_connector:
                            all_connectors.append(clean_connector)

    return all_connectors


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    corpus_folder = os.path.join(os.getcwd(), "corpus15K")
    output_file = "connect.txt"

    # 1. SCAN
    connectors_list = scan_connectors(corpus_folder)

    # 2. COMPTAGE
    counts = Counter(connectors_list)
    sorted_by_freq = counts.most_common()

    # 3. GÉNÉRATION REGEX (Tri par longueur décroissante pour la sécurité)
    unique_connectors = sorted(counts.keys(), key=len, reverse=True)
    escaped_connectors = [re.escape(c) for c in unique_connectors]
    regex_pattern = f"^(.*?)\\s+({'|'.join(escaped_connectors)})\\s+(.*)$"

    # 4. ÉCRITURE DANS LE FICHIER
    print(f"💾 Sauvegarde dans '{output_file}'...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# FICHIER GÉNÉRÉ AUTOMATIQUEMENT\n")
        f.write("# Format: connecteur;frequence\n")
        f.write("# Vous pouvez supprimer manuellement les lignes erronées ici.\n\n")

        # Écriture de la liste
        f.write("--- LISTE DES CONNECTEURS ---\n")
        for conn, count in sorted_by_freq:
            f.write(f"{conn};{count}\n")

        # Écriture de la Regex
        f.write("\n--- REGEX SUGGÉRÉE (A COPIER DANS PYTHON) ---\n")
        f.write(f'r"{regex_pattern}"\n')

    print("✅ Terminé ! Ouvrez 'connect.txt' pour voir le résultat.")