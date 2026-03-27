# Puls-Events — Chatbot RAG

POC d'un chatbot intelligent basé sur le **Retrieval-Augmented Generation (RAG)** pour recommander des événements culturels.

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Orchestrateur RAG | LangChain |
| Modèle NLP | Mistral (via API) |
| Base vectorielle | Faiss (faiss-cpu) |
| API REST | FastAPI |
| Source de données | API Open Agenda |

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd pull-events
```

### 2. Créer un environnement virtuel

```bash
python -m venv env
source env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env et renseigner votre clé API Mistral
```

### 5. Vérifier l'installation

```bash
python scripts/check_imports.py
```

## Structure du projet

```
pull-events/
├── docs/                  # Documentation et template rapport
├── scripts/               # Scripts utilitaires
│   └── check_imports.py   # Vérification de l'environnement
├── .env.example           # Template des variables d'environnement
├── .gitignore
├── README.md
└── requirements.txt
```
