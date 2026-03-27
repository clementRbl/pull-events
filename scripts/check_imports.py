"""
Vérification de l'environnement de développement.

Vérifie que toutes les dépendances clés du projet Puls-Events
sont correctement installées et importables.
"""

import sys


def verifier_import(nom_module: str, nom_package: str = None) -> bool:
    """Tente d'importer un module et affiche le résultat.

    Args:
        nom_module: Nom du module Python à importer.
        nom_package: Nom du package pip correspondant (pour affichage).

    Returns:
        True si l'import réussit, False sinon.
    """
    nom_affiche = nom_package or nom_module
    try:
        __import__(nom_module)
        print(f"  [OK] {nom_affiche}")
        return True
    except ImportError as e:
        print(f"  [ERREUR] {nom_affiche} — {e}")
        return False


def main():
    """Point d'entrée : vérifie tous les imports requis."""
    print("Vérification de l'environnement Puls-Events")

    # Liste des modules à vérifier :
    modules = [
        ("langchain", "langchain"),
        ("langchain_mistralai", "langchain-mistralai"),
        ("langchain_community", "langchain-community"),
        ("faiss", "faiss-cpu"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pandas", "pandas"),
        ("requests", "requests"),
        ("dotenv", "python-dotenv"),
    ]

    print("\nVérification des imports :\n")
    erreurs = 0
    for nom_module, nom_package in modules:
        if not verifier_import(nom_module, nom_package):
            erreurs += 1

    # Résumé final
    if erreurs == 0:
        print("Tous les imports sont OK. Environnement prêt.")
    else:
        print(f"{erreurs} erreur(s) détectée(s).")
        print("Installez les dépendances manquantes :")
        print("  pip install -r requirements.txt")

    sys.exit(erreurs)


if __name__ == "__main__":
    main()
