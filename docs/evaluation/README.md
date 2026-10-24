# Évaluation avant/après : Faiss → pgvector → Azure

Mesures des 24 et 25 octobre 2026, sur la même base de 39 089 chunks (37 338 événements Open Agenda d'Île-de-France).

- **Faiss → pgvector** (en local) : **les mêmes vecteurs**, copiés tels quels de Faiss vers pgvector (`scripts/faiss_baseline.py export`). Seule la base vectorielle change.
- **Local → Azure, Mistral (colonne A)** : la table pgvector locale copiée telle quelle (`pg_dump`) sur PostgreSQL Flexible Server. Seule l'infrastructure change.
- **Azure OpenAI (colonne B)** : LLM `gpt-4.1-mini` et embeddings `text-embedding-3-small`, donc toute la base réindexée. Le modèle et la recherche changent ensemble.

## Protocole

| Élément | Valeur |
|---|---|
| Jeu de test | `data/test_questions.json`, 20 questions annotées (dont 1 hors sujet) |
| LLM du RAG | `ministral-8b-latest` (Azure B : `gpt-4.1-mini`, déploiement Standard régional France Central), température 0,3, top_k = 5 |
| Embeddings | `mistral-embed`, 1024 dimensions (Azure B : `text-embedding-3-small`, 1536 dimensions, Data Zone UE) |
| Juge Ragas | `claude-haiku-5-5` (Anthropic) : un autre fournisseur que les modèles évalués. Ses embeddings (answer relevancy) restent `mistral-embed` dans toutes les colonnes |
| Faiss | `IndexFlatL2`, recherche exacte, en mémoire |
| pgvector | PostgreSQL 17, pgvector 0.8.7, index HNSW (`m=16`, `ef_construction=64`), distance cosinus. En local : conteneur Docker. Sur Azure : Flexible Server Burstable B1ms (1 vCore, 2 Go) |
| Commande | `uv run python -m scripts.evaluate_rag --backend {faiss,pgvector} --output-label <nom>` |
| Où tourne le RAG évalué | Sur le PC (Paris). Colonnes Azure : le script interroge la base Azure (aller-retour réseau : 6 ms) et les modèles Azure ; la latence de l'API déployée est mesurée à part |

Les fichiers `results_*.json` contiennent le résumé et le détail par question (réponse, sources, scores, tokens, coût, latences).

## Résultats

| | Faiss (v1) | pgvector, `ef_search=40` | pgvector, `ef_search=100` | **Azure A** : Mistral | **Azure B** : Azure OpenAI |
|---|---|---|---|---|---|
| LLM / embeddings | Ministral 8B / mistral-embed | idem | idem | idem | gpt-4.1-mini / text-embedding-3-small |
| Base | Faiss en mémoire | pgvector local | pgvector local | pgvector Azure B1ms | pgvector Azure B1ms |
| **Faithfulness** | 0,871 | 0,843 | 0,884 | 0,878 | **0,932** |
| **Answer relevancy** | 0,826 | 0,837 | 0,831 | 0,843 | **0,878** |
| **Context precision** | 0,582 | 0,518 | 0,518 | 0,527 | **0,707** |
| Mêmes 5 sources que Faiss | — | 16 / 20 | 18 / 20 | 18 / 20 | 0 / 20 (autres embeddings) |
| Embedding de la question (moyenne) | ~0,4 s | ~0,4 s | 0,37 s | 0,46 s | **0,08 s** |
| Recherche vectorielle (moyenne / médiane) | 14,8 ms | 4,3 ms | 5,6 ms | 832 ms / 622 ms | 206 ms / 141 ms |
| Génération (moyenne) | — | — | 3,10 s | 2,60 s | 2,59 s |
| Latence totale moyenne | 4,31 s | 3,06 s | 3,47 s | 3,89 s | **2,87 s** |
| Latence totale p95 | 8,18 s | 4,38 s | 6,16 s | 7,08 s | 4,49 s |
| Tokens par question (entrée / sortie) | 1 390 / 323 | 1 342 / 372 | 1 392 / 332 | 1 392 / 358 | 1 318 / 311 |
| **Coût LLM par question** | 0,000171 $ | 0,000171 $ | 0,000172 $ | 0,000175 $ | 0,00124 $ |
| **Coût pour 1 000 questions** | 0,17 $ | 0,17 $ | 0,17 $ | 0,18 $ | **1,24 $** |

Latence de l'API déployée sur Azure (configuration B), mesurée en HTTP depuis un client à Paris sur les 20 questions (une toutes les 7 s, pour rester sous la limite de débit) : **moyenne 2,63 s, médiane 2,50 s, p95 4,19 s**, 20 réponses sur 20 en HTTP 200. C'est un peu moins que la colonne B mesurée depuis le PC (2,87 s), car le conteneur est dans le même centre de données que la base et le modèle.

Prix utilisés : 0,10 $ par million de tokens en entrée et en sortie pour Ministral 8B (tarif relevé sur des comparateurs, non vérifié sur la page officielle de Mistral). La latence totale comprend l'embedding de la question (~0,4 s), la recherche et la génération (~2,7 à 3,8 s).

## Ce qu'on peut en conclure

**La qualité ne bouge pas de façon mesurable.** Sur les 16 questions où Faiss et pgvector (`ef_search=40`) ont renvoyé exactement les mêmes chunks, les scores du juge varient déjà en moyenne de 0,06 (faithfulness), 0,03 (answer relevancy) et 0,17 (context precision) d'une exécution à l'autre. Les écarts du tableau sont de cet ordre : c'est du bruit, pas un effet de la base.

**La recherche est 3 fois plus rapide**, mais cela ne change rien pour l'utilisateur : 10 ms gagnées sur plus de 3 secondes. Les écarts de latence totale viennent surtout de la génération par l'API Mistral, qui varie d'un appel à l'autre.

**Le coût par réponse est identique**, et c'est attendu : il dépend du LLM et du nombre de tokens envoyés, pas de la base vectorielle. Environ 0,17 $ pour 1 000 questions avec Ministral 8B.

## Ce que change Azure

**L'infrastructure seule ne change rien à la qualité (colonne A).** Mêmes vecteurs, même LLM : les 5 sources sont identiques à celles du pgvector local pour les 20 questions, et les scores restent dans le bruit du juge.

**Azure OpenAI améliore les trois scores (colonne B)**, et pas seulement en moyenne. Sur les 20 questions comparées à la colonne A, à plus de 0,05 près :

| | B meilleure | B moins bonne |
|---|---|---|
| Faithfulness | 11 questions | 4 |
| Answer relevancy | 7 | 0 |
| Context precision | 9 | 2 |

La hausse de context precision (+0,18) vient de la recherche : les embeddings `text-embedding-3-small` ne renvoient aucune des listes de 5 sources de `mistral-embed`, et le juge trouve leurs chunks plus pertinents. Le modèle et les embeddings changent ensemble : impossible, avec ces deux colonnes, d'attribuer le gain à l'un ou à l'autre. Une seule exécution par configuration : à confirmer sur plusieurs passages.

**La qualité coûte 7 fois plus cher** : 1,24 $ pour 1 000 questions avec `gpt-4.1-mini` (0,484 $ / 1,936 $ par million de tokens en déploiement régional), contre 0,18 $ avec Ministral 8B. Cela reste moins d'un dixième de centime par réponse.

**L'embedding de la question passe de 0,46 s à 0,08 s** : le déploiement est en Europe, à quelques millisecondes, alors que l'API Mistral publique ajoute sa propre file d'attente.

**Le petit serveur PostgreSQL est le point faible.** La recherche passe de 5 ms en local à 140-620 ms (médiane) sur Azure, alors que l'aller-retour réseau n'est que de 6 ms. Le serveur B1ms (2 Go de RAM, `shared_buffers` de 256 Mo) ne garde pas les index HNSW en mémoire : la première série de requêtes sur la table Mistral lit le disque (2,2 s pour la première question, 0,14 s pour les dernières). Pistes avant une vraie production : un serveur à 4 Go de RAM ou plus, une seule table au lieu de deux, ou des vecteurs en demi-précision (`halfvec`) pour diviser l'index par deux. Pour la démo, ce coût reste petit devant les 2,6 s de génération.

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

Chaque évaluation de 20 questions coûte environ 0,12 $ de juge, estimation affichée par le script avant de lancer le juge (option `--max-questions` pour en évaluer moins). Les deux colonnes Azure ont coûté environ 0,25 $ de juge, 0,03 $ d'Azure OpenAI et moins de 0,01 $ de Mistral ; la réindexation complète avec `text-embedding-3-small`, environ 0,3 $.

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
- Colonnes Azure : le RAG tourne sur le PC et interroge les ressources Azure. La latence vue par un utilisateur est celle de l'API déployée, mesurée à part en HTTP.
- La colonne B change le LLM et les embeddings en même temps : une colonne intermédiaire (Ministral avec `text-embedding-3-small`) séparerait les deux effets.
- Le modèle du RAG a dû passer de `mistral-small-latest` (v1) à `ministral-8b-latest` : la clé API utilisée n'a pas de quota sur les modèles Small et Medium. Les trois colonnes utilisent le même modèle, la comparaison reste valable.
