---
title: "De Faiss à pgvector : ce que coûte vraiment une réponse de mon chatbot RAG"
date: 2026-10-24
tags: [RAG, pgvector, PostgreSQL, Langfuse, LLM, observabilité]
status: brouillon
---

# De Faiss à pgvector : ce que coûte vraiment une réponse de mon chatbot RAG

Puls-Events est un chatbot qui recommande des sorties culturelles en Île-de-France. Il interroge 37 000 événements Open Agenda, découpés en 39 089 morceaux de texte (des « chunks ») et vectorisés avec Mistral. Dans sa première version, ces vecteurs vivaient dans un index Faiss : un fichier sur disque, chargé en mémoire au démarrage de l'API.

Ça marchait. Mais je ne savais répondre à aucune des questions qu'on pose avant de mettre un système comme celui-ci en production : combien coûte une réponse ? Combien de temps prend-elle, et où passe ce temps ? Et si je change de base vectorielle, est-ce que la qualité baisse ?

J'ai donc refait deux choses : la base vectorielle, passée à PostgreSQL + pgvector, et l'observabilité, avec Langfuse. Puis j'ai mesuré, avant et après.

## Pourquoi quitter Faiss

Faiss est excellent pour chercher vite dans des vecteurs. Mais en production, un fichier pickle chargé en mémoire pose trois problèmes concrets :

- **Reconstruire l'index, c'est réécrire le fichier.** Il faut redémarrer l'API ou gérer soi-même le rechargement.
- **Les métadonnées sont enfermées dans un blob Python.** Impossible de filtrer « événements à Versailles en décembre » sans tout charger.
- **On ne l'inspecte qu'en Python.** Pas de client SQL, pas de sauvegarde standard, pas de base managée chez un hébergeur cloud.

Avec pgvector, les chunks deviennent une table PostgreSQL ordinaire : une colonne `embedding vector(1024)` indexée en HNSW, et une colonne par métadonnée (titre, ville, dates, URL). Une sauvegarde `pg_dump` suffit, et le même code tournera demain sur un PostgreSQL managé.

## Reconstruire sans couper le service

Réindexer les 39 089 chunks prend une quarantaine de minutes, à cause du débit de l'API d'embeddings. Je ne voulais pas que le chatbot réponde « base indisponible » pendant ce temps.

La solution tient en une transaction. L'indexation écrit dans une table de travail, `event_chunks_staging`. Une fois la table remplie et indexée, je la mets en service :

```sql
BEGIN;
DROP TABLE IF EXISTS event_chunks;
ALTER TABLE event_chunks_staging RENAME TO event_chunks;
ALTER INDEX event_chunks_staging_pkey RENAME TO event_chunks_pkey;
ALTER INDEX event_chunks_staging_hnsw RENAME TO event_chunks_hnsw;
COMMIT;
```

Les deux dernières lignes ne sont pas décoratives. Sans elles, la reconstruction suivante crée une nouvelle table de travail dont la clé primaire porte un nom déjà pris, et elle échoue. Un test d'intégration lance deux reconstructions d'affilée sur une vraie base pour vérifier ce point.

## Une trace par question avec Langfuse

Langfuse est une plateforme open source d'observabilité pour les applications LLM. Je l'héberge moi-même dans le même `docker-compose.yml` que la base. Chaque appel à `/ask` produit une trace :

```
ask
├── retrieval       titres des chunks, durée de l'embedding, durée de la recherche
└── mistral-chat    prompt complet, réponse, tokens d'entrée et de sortie, coût en $
```

J'ai préféré instrumenter explicitement plutôt que de brancher le callback LangChain générique. Le coût est calculé dans l'application à partir des tokens renvoyés par Mistral, puis envoyé à Langfuse. Le script d'évaluation et le tableau de bord utilisent donc la même grille de prix, et je sais d'où vient chaque chiffre.

Sans les clés Langfuse dans l'environnement, l'API fonctionne normalement et ne trace rien. Un service d'observabilité en panne ne doit pas faire tomber le produit.

## Le protocole de mesure

Pour que la comparaison ait un sens, une seule chose change entre « avant » et « après » :

- **mêmes vecteurs** : les embeddings calculés pour Faiss ont été copiés tels quels dans pgvector, sans repasser par l'API ;
- **même LLM** pour répondre (Ministral 8B) ;
- **même jeu de test** : 20 questions annotées, dont une volontairement hors sujet (« Quel temps fera-t-il demain ? ») ;
- **même juge** : Ragas note la fidélité au contexte, la pertinence de la réponse et la précision des chunks récupérés. Le juge est Claude Haiku, d'un autre fournisseur que le modèle évalué, pour qu'aucun modèle ne note ses propres réponses.

## Les résultats

| | Faiss | pgvector (`ef_search=40`) | pgvector (`ef_search=100`) |
|---|---|---|---|
| Faithfulness | 0,871 | 0,843 | 0,884 |
| Answer relevancy | 0,826 | 0,837 | 0,831 |
| Context precision | 0,582 | 0,518 | 0,518 |
| Recherche vectorielle | 14,8 ms | 4,3 ms | 5,6 ms |
| Latence totale moyenne | 4,3 s | 3,1 s | 3,5 s |
| Coût pour 1 000 questions | 0,17 $ | 0,17 $ | 0,17 $ |

### La qualité ne bouge pas, et il faut le prouver

Faithfulness 0,87 contre 0,88 : est-ce une amélioration ? Non. Sur les 16 questions où Faiss et pgvector ont renvoyé exactement les mêmes chunks, le juge donne déjà des notes différentes d'une exécution à l'autre : 0,06 d'écart moyen en fidélité, et jusqu'à 0,17 en précision du contexte. Tout écart plus petit que ce bruit ne veut rien dire.

C'est la leçon la plus utile de l'exercice : avant de comparer deux versions, il faut mesurer le bruit de sa propre évaluation.

### Le coût ne dépend pas de la base

0,17 $ pour 1 000 questions dans les trois cas. C'est logique : le coût vient du LLM et du nombre de tokens envoyés (environ 1 400 en entrée, 5 chunks plus le prompt, et 350 en sortie). Pour le réduire, le levier est le nombre et la taille des chunks, pas la base vectorielle.

### La recherche est plus rapide, mais ce n'est pas le sujet

pgvector cherche en 4 à 6 ms contre 15 ms pour Faiss, qui parcourt les 39 089 vecteurs un par un. L'utilisateur ne le verra jamais : la génération par le LLM prend 3 secondes. Les écarts de latence totale du tableau viennent de l'API Mistral, pas de la base.

## Le réglage que je n'aurais pas vu sans mesurer

HNSW est un index approché : il ne garantit pas de trouver les vrais plus proches voisins. Avec le réglage par défaut de pgvector, seules 16 questions sur 20 récupéraient exactement les mêmes chunks que la recherche exacte de Faiss.

J'ai mesuré le rappel@5, c'est-à-dire la part des 5 meilleurs résultats exacts que l'index retrouve :

| Recherche | Rappel@5 | Temps |
|---|---|---|
| Exacte (parcours de toute la table) | 1,00 | 113 ms |
| HNSW, `ef_search=40` (défaut) | 0,91 | 1,7 ms |
| HNSW, `ef_search=100` | 0,99 | 2,1 ms |

`ef_search` est le nombre de candidats que l'index explore avant de choisir. À 40, un chunk sur onze n'est pas le bon. À 100, il n'en manque presque plus, pour 0,4 milliseconde de plus. C'est maintenant la valeur par défaut, et une variable d'environnement.

Au passage, un piège qui m'a fait douter de mes chiffres : psycopg prépare automatiquement une requête après cinq exécutions et réutilise ensuite son plan. Mon premier script de mesure affichait un rappel parfait en 111 ms. HNSW n'y était pour rien : le plan « recherche exacte » préparé au début servait pour toutes les mesures.

## Ce que coûte l'évaluation

Dernière surprise, révélée par Langfuse : le SDK d'Anthropic émet ses propres traces OpenTelemetry, et Langfuse les a enregistrées avec celles du chatbot. J'ai donc pu chiffrer le juge.

Noter 20 réponses coûte environ 0,12 $. Produire ces 20 réponses coûte 0,0034 $. **L'évaluation coûte 35 fois plus cher que le système évalué.** Avec un modèle de génération aussi bon marché, ce n'est pas absurde, mais il faut le savoir avant de lancer Ragas sur chaque commit.

## Deux bugs trouvés en route

- **Deux recherches par question.** La v1 cherchait une fois les chunks pour afficher les sources, puis une seconde fois dans la chaîne LangChain. Il n'y a plus qu'une recherche, chronométrée.
- **Des lettres décoratives qui coûtent cher.** Certaines descriptions contiennent du texte « gras » copié des réseaux sociaux (𝗣𝗥𝗢𝗚𝗥𝗔𝗠𝗠𝗘). Ce sont des caractères Unicode mathématiques, et chacun coûte plusieurs tokens. Un chunk de 4 306 caractères atteignait 9 138 tokens, au-delà de la limite de 8 192 de l'API d'embeddings, et l'indexation s'arrêtait à mi-chemin. Une normalisation Unicode NFKC le ramène à 1 709 tokens.

## Et ensuite

La configuration (base, modèles, prix, `ef_search`) passe entièrement par des variables d'environnement. La prochaine étape est de redéployer ce même service sur Azure : PostgreSQL managé avec pgvector, Azure OpenAI pour le LLM, Container Apps pour l'API, le tout décrit en Terraform. Le tableau ci-dessus servira de référence.

Le code est sur GitHub : [clementRbl/pull-events](https://github.com/clementRbl/pull-events).
