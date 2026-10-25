# Préparer AI-103 à partir de pull-events

**AI-102 n'existe plus.** L'examen a été retiré le 30 juin 2026 ([guide d'étude AI-102](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-102)). Son successeur est **AI-103 « Developing AI Apps and Agents on Azure »**, qui mène à la certification **Microsoft Certified: Azure AI Apps and Agents Developer Associate**.

Programme vérifié le 24 octobre 2026 sur le [guide d'étude officiel AI-103](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103) (compétences mesurées au 16 avril 2026).

## L'examen en bref

| | |
|---|---|
| Durée | 120 minutes, surveillé (Pearson VUE) |
| Score pour réussir | 700 sur 1 000 |
| Langues | dont le **français** |
| Prérequis implicite | Python, Microsoft Foundry |
| Prix en France | non affiché sur Microsoft Learn : à lire au moment de la réservation |
| Validité | un an, renouvelable par un test en ligne gratuit |
| Entraînement | [examen d'entraînement](https://aiskillsnavigator.microsoft.com/credentials/cert-3fb198f57997226a824aa5f52a1a22af9a4597941b2288ed39371c7a9e6bd7c9) (connexion requise), [bac à sable de l'interface](https://aka.ms/examdemo) |

Par rapport à AI-102, le centre de gravité a bougé : l'IA générative et les agents pèsent 30 à 35 %, contre 20 à 30 % auparavant. La vision et le langage « classiques » (Custom Vision, LUIS, question answering) ont laissé place aux modèles multimodaux et à Content Understanding.

Légende : ✅ fait et démontrable dans un dépôt, 🟡 partiel ou fait hors Azure, ❌ à apprendre.

## A. Planifier et gérer une solution Azure AI (25-30 %)

| Compétence | Statut | Ce que tu as fait / ce qui manque |
|---|---|---|
| Choisir un modèle selon la tâche (LLM, petit modèle, multimodal) | ✅ | `gpt-4.1-mini` choisi parce que sans raisonnement, comme Ministral 8B ; Mistral sur Foundry écarté (seuls Large 3 et Medium 3.5 y sont vendus par Azure) |
| Choisir le service de récupération et d'indexation | ✅ | Comparaison pgvector / Azure AI Search ([deploiement.md](deploiement.md)) ; réglage HNSW `ef_search` avec mesure du rappel ([évaluation](../evaluation/README.md)) |
| Choisir mémoire, outils et connaissances pour un agent | 🟡 | Fait avec LangGraph (`ffe-chess-openings-agent`, `hermes`), pas avec les services Foundry |
| Concevoir l'infrastructure | ✅ | Container Apps + PostgreSQL Flexible + Key Vault + Foundry |
| Choisir le type de déploiement | ✅ | Standard régional, Global Standard, Data Zone : vu concrètement (quota Data Zone à 0 pour `gpt-4.1-mini`, d'où le déploiement régional) |
| Configurer les déploiements de modèles | ✅ | Capacité en milliers de tokens par minute, version de modèle figée |
| Intégrer un projet Foundry à la CI/CD | ❌ | Prévu à l'étape Terraform. À voir : GitHub Actions + `azure/login` + déploiement de modèles en code |
| Quotas, limites de débit, coûts | ✅ | Quotas lus avec `az cognitiveservices usage list`, alerte budget, coût par question dans Langfuse |
| Surveiller performance, dérive, sécurité, ancrage | 🟡 | Ancrage mesuré par Ragas (faithfulness) ; dérive suivie dans `credit-scoring-mlops` (Evidently). Manque : Azure Monitor, Application Insights, évaluations continues dans Foundry |
| Surveiller l'ingestion et la santé d'un index de recherche | ❌ | Lié à Azure AI Search, non utilisé |
| Sécurité : identité managée, réseau privé, sans clé, rôles | 🟡 | Identité managée de Container Apps pour lire Key Vault, Key Vault en mode RBAC. Manque : accès **sans clé** à Azure OpenAI (rôle `Cognitive Services OpenAI User` + `DefaultAzureCredential`), points de terminaison privés |
| Filtres de sécurité, garde-fous, modération | 🟡 | Filtre de contenu par défaut appliqué au déploiement, sans personnalisation. À voir : filtres personnalisés, listes de blocage, Prompt Shields |
| Instrumentation responsable (évaluateurs, évaluations de sécurité) | 🟡 | Évaluateurs Ragas avec juge externe. Manque : évaluateurs intégrés de Foundry (groundedness, safety) |
| Audit : traces, provenance, approbation | ✅ | Traces Langfuse par question, sources renvoyées par l'API |
| Gouverner un agent (supervision, contraintes, accès aux outils) | ❌ | Agent Service, approbation humaine des appels d'outils |

## B. Solutions génératives et agents (30-35 %), le plus gros bloc

| Compétence | Statut | Ce que tu as fait / ce qui manque |
|---|---|---|
| Déployer et consommer des LLM | ✅ | Azure OpenAI via `langchain-openai`, Mistral via API |
| Implémenter un RAG | ✅ | `pull-events` entier |
| Workflows, outils, raisonnement multi-étapes | 🟡 | LangGraph (`ffe-chess-openings-agent`) ; pas de Foundry |
| Évaluer : fabrications, pertinence, qualité, sécurité | ✅ | Ragas (faithfulness, answer relevancy, context precision) sur 20 questions, juge d'un autre fournisseur |
| Intégrer avec les SDK Foundry et connecter une application à un projet Foundry | ❌ | **Priorité.** Tu as utilisé une ressource Foundry mais pas de *projet* Foundry ni le SDK `azure-ai-projects` |
| Construire un agent avec Foundry Agent Service | ❌ | **Priorité.** Rôle, instructions, outils (recherche, fonctions), fils de conversation |
| Agents avec récupération, appels de fonctions, mémoire | 🟡 | Fait avec LangGraph et Milvus, à transposer dans Agent Service |
| Solutions multi-agents orchestrées | ❌ | Microsoft Agent Framework |
| Flux autonomes avec garde-fous et approbation | ❌ | |
| Surveiller et évaluer un agent déployé | 🟡 | `saadflow` : évaluations d'agents en CI, hors Azure |
| Prompt engineering, paramètres de génération | ✅ | Prompt système strict, température 0,3 |
| Réflexion, auto-critique | 🟡 | `veille-ia` : double contrôle des sorties |
| Observabilité : traces, tokens, latences | ✅ | Langfuse : latence par étape, tokens, coût par question |
| Orchestrer plusieurs modèles | 🟡 | Juge Claude + RAG Mistral/OpenAI ; LLM multi-fournisseurs dans `saadflow` |

## C. Vision (10-15 %)

Rien de fait. À apprendre :
- génération et édition d'images et de vidéos (gpt-image, Sora sur Foundry), inpainting par masque ;
- compréhension multimodale : légendes, texte alternatif, questions sur une image ;
- **Azure Content Understanding** pour l'analyse d'images et de vidéos (modes simple et « pro ») ;
- filtrage des contenus visuels, injection de prompt cachée dans le texte d'une image.

## D. Analyse de texte et parole (10-15 %)

| Compétence | Statut | Ce que tu as fait / ce qui manque |
|---|---|---|
| Extraire entités, résumés, JSON structuré | 🟡 | Sorties structurées faites ailleurs ; pas avec les Foundry Tools (Language) |
| Sentiment, ton, contenus sensibles (PII) | 🟡 | Anonymisation RGPD dans `agent-ia-triage-medical`, sans service Azure |
| Traduction (Azure Translator ou LLM) | ❌ | |
| Parole : speech-to-text, text-to-speech, traduction, parole comme modalité d'agent | ❌ | Azure AI Speech |

## E. Extraction d'information (10-15 %)

| Compétence | Statut | Ce que tu as fait / ce qui manque |
|---|---|---|
| Ingérer et indexer du contenu | ✅ (texte) | Pipeline Open Agenda → chunks → embeddings → pgvector |
| Recherche sémantique, hybride et vectorielle | 🟡 | Vectorielle seulement. À apprendre dans **Azure AI Search** : hybride (BM25 + vecteurs), *semantic ranker*, profils vectoriels |
| Enrichissement par compétences (skillsets) | ❌ | Indexeurs, skillsets intégrés et personnalisés |
| Ingestion RAG avec OCR | ❌ | |
| Brancher la récupération sur des agents | 🟡 | Fait avec LangGraph ; à refaire avec l'outil Azure AI Search d'Agent Service |
| Extraction de documents (OCR, mise en page, champs) | ❌ | Content Understanding, Document Intelligence |

## Ce qui reste à réviser, par ordre de rentabilité

1. **Foundry côté code (B, 30-35 %).** Créer un *projet* Foundry, s'y connecter avec `azure-ai-projects`, puis construire un agent avec **Agent Service** qui interroge `pull-events` comme outil. C'est le plus gros bloc de l'examen et le plus vendeur en mission.
2. **Azure AI Search (A et E).** Recréer l'index `pull-events` sur le niveau Free avec un sous-ensemble (50 Mo maximum), tester recherche hybride et *semantic ranker*, un indexeur et un skillset.
3. **Sécurité sans clé (A).** Remplacer `AZURE_OPENAI_API_KEY` par l'identité managée de Container Apps (rôle `Cognitive Services OpenAI User`). Petit changement, question classique d'examen.
4. **IA responsable (A).** Filtres de contenu personnalisés, Prompt Shields, évaluateurs de sécurité de Foundry.
5. **Content Understanding et Document Intelligence (C et E).** Un exercice sur des PDF et des images.
6. **Speech, Translator, Language (D).** Survol avec les modules Microsoft Learn ; peu de temps, peu de poids chacun.
7. **Agent Framework et multi-agents (B).** Après Agent Service.

Avant de réserver : faire l'examen d'entraînement officiel, viser 80 % de manière régulière.

## Points non vérifiés

- Le prix en euros (affiché seulement pendant la réservation Pearson VUE).
- La liste des parcours Microsoft Learn associés : la section « Prepare for the exam » de la [page de certification](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-apps-and-agents-developer-associate/) ne s'est pas affichée sans navigateur.
- L'existence d'une passerelle pour les titulaires d'AI-102 : aucune mention sur les pages officielles lues.
