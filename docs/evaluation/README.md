# Évaluation avant/après : Faiss → PostgreSQL + pgvector

Mesures du 24 octobre 2026, sur la même base de 39 089 chunks (37 338 événements Open Agenda d'Île-de-France), avec **les mêmes vecteurs** : les embeddings calculés pour Faiss ont été copiés tels quels dans pgvector (`scripts/faiss_baseline.py export`). Seule la base vectorielle change.

## Protocole

| Élément | Valeur |
|---|---|
| Jeu de test | `data/test_questions.json`, 20 questions annotées (dont 1 hors sujet) |
| LLM du RAG | `ministral-8b-latest`, température 0,3, top_k = 5 |
| Embeddings | `mistral-embed` (1024 dimensions) |
| Juge Ragas | `claude-haiku-5-5` (Anthropic) : un autre fournisseur que le modèle évalué |
| Faiss | `IndexFlatL2`, recherche exacte, en mémoire |
| pgvector | PostgreSQL 17, pgvector 0.8.7, index HNSW (`m=16`, `ef_construction=64`), distance cosinus |
| Commande | `uv run python -m scripts.evaluate_rag --backend {faiss,pgvector}` |

Les fichiers `results_*.json` contiennent le résumé et le détail par question (réponse, sources, scores, tokens, coût, latences).

## Résultats

| | Faiss (v1) | pgvector, `ef_search=40` | pgvector, `ef_search=100` |
|---|---|---|---|
| **Faithfulness** | 0,871 | 0,843 | 0,884 |
| **Answer relevancy** | 0,826 | 0,837 | 0,831 |
| **Context precision** | 0,582 | 0,518 | 0,518 |
| Mêmes 5 sources que Faiss | — | 16 / 20 | 18 / 20 |
| Recherche vectorielle (moyenne) | 14,8 ms | 4,3 ms | 5,6 ms |
| Latence totale moyenne | 4,31 s | 3,06 s | 3,47 s |
| Latence totale p95 | 8,18 s | 4,38 s | 6,16 s |
| Tokens par question (entrée / sortie) | 1 390 / 323 | 1 342 / 372 | 1 392 / 332 |
| **Coût LLM par question** | 0,000171 $ | 0,000171 $ | 0,000172 $ |
| **Coût pour 1 000 questions** | 0,17 $ | 0,17 $ | 0,17 $ |

Prix utilisés : 0,10 $ par million de tokens en entrée et en sortie pour Ministral 8B (tarif relevé sur des comparateurs, non vérifié sur la page officielle de Mistral). La latence totale comprend l'embedding de la question (~0,4 s), la recherche et la génération (~2,7 à 3,8 s).

## Ce qu'on peut en conclure

**La qualité ne bouge pas de façon mesurable.** Sur les 16 questions où Faiss et pgvector (`ef_search=40`) ont renvoyé exactement les mêmes chunks, les scores du juge varient déjà en moyenne de 0,06 (faithfulness), 0,03 (answer relevancy) et 0,17 (context precision) d'une exécution à l'autre. Les écarts du tableau sont de cet ordre : c'est du bruit, pas un effet de la base.

**La recherche est 3 fois plus rapide**, mais cela ne change rien pour l'utilisateur : 10 ms gagnées sur plus de 3 secondes. Les écarts de latence totale viennent surtout de la génération par l'API Mistral, qui varie d'un appel à l'autre.

**Le coût par réponse est identique**, et c'est attendu : il dépend du LLM et du nombre de tokens envoyés, pas de la base vectorielle. Environ 0,17 $ pour 1 000 questions avec Ministral 8B.

## Le réglage qui compte : `hnsw.ef_search`

HNSW est un index **approché** : il ne garantit pas de trouver les vrais plus proches voisins. Mesure du rappel@5 (part des 5 résultats de la recherche exacte retrouvés par l'index) sur les 20 questions :

| Recherche | Rappel@5 | Temps de la requête SQL |
|---|---|---|
| Exacte (parcours complet de la table) | 1,00 | 113 ms |
| HNSW, `ef_search=40` (défaut pgvector) | 0,91 | 1,7 ms |
| HNSW, `ef_search=100` | 0,99 | 2,1 ms |

Avec le réglage par défaut, environ 1 chunk sur 11 n'est pas celui qu'une recherche exacte aurait renvoyé. À 100, le rappel est quasi parfait pour 0,4 ms de plus. L'application utilise donc `HNSW_EF_SEARCH=100` par défaut.

Piège rencontré pendant la mesure : psycopg prépare automatiquement une requête après 5 exécutions et réutilise son plan. Un plan préparé pendant la mesure « recherche exacte » (index désactivé) était réutilisé pour les mesures HNSW, qui affichaient alors un rappel parfait… en 111 ms. Il faut `prepare=False` pour comparer des plans différents.

## Coût de l'évaluation elle-même

Langfuse a aussi enregistré les appels du juge Claude (le SDK Anthropic émet des traces OpenTelemetry, que le client Langfuse récupère). Trois évaluations ont produit 601 appels au juge, environ 1,0 million de tokens en entrée et 0,5 million en sortie (raisonnement compris) : **environ 0,12 $ par évaluation de 20 questions**, contre **0,0034 $** pour que le RAG réponde à ces mêmes 20 questions. Évaluer coûte ici 35 fois plus cher que répondre.

## Stockage et mémoire

| | Faiss | pgvector |
|---|---|---|
| Données | 201 Mo sur disque (index + pickle), chargés en RAM par l'API | table 251 Mo + index HNSW 303 Mo |
| Mémoire du conteneur | dans le processus de l'API | PostgreSQL : ~180 Mo |
| Construction de l'index | pendant l'indexation | HNSW : 15 s pour 39 089 vecteurs |

Langfuse auto-hébergé (web, worker, ClickHouse, PostgreSQL, Redis, MinIO) occupe environ 2,2 Go de RAM au repos.

## Limites

- 20 questions avec des références génériques : suffisant pour comparer deux versions, pas pour une mesure absolue de qualité.
- La context precision est très instable avec ce juge : sur une même question et les mêmes chunks, elle passe parfois de 0 à 1 entre deux exécutions.
- Une seule exécution par configuration ; une moyenne sur plusieurs exécutions réduirait le bruit.
- Le modèle du RAG a dû passer de `mistral-small-latest` (v1) à `ministral-8b-latest` : la clé API utilisée n'a pas de quota sur les modèles Small et Medium. Les trois colonnes utilisent le même modèle, la comparaison reste valable.
