# Déploiement sur Azure

Déploiement manuel, avec la CLI `az`, réalisé le 24 octobre 2026. Ce document est l'inventaire exact de ce qui a été créé et de ses réglages. Depuis le 25 octobre, la même infrastructure est décrite et gérée par Terraform ([terraform.md](terraform.md)) et l'API tourne aussi sur Kubernetes ([kubernetes.md](kubernetes.md)) : les valeurs ci-dessous restent la référence historique, les valeurs courantes sont dans `infra/terraform/`.

## Architecture

```
Internet ──HTTPS──▶ Container Apps (ca-pull-events-api, image GHCR)
                      │  identité managée ──lit──▶ Key Vault (secrets)
                      ├──▶ PostgreSQL Flexible Server + pgvector (France Central)
                      ├──▶ Foundry / Azure OpenAI (gpt-4.1-mini, text-embedding-3-small)
                      └──▶ Langfuse Cloud (UE) : traces, tokens, coût
```

URL de l'API : https://ca-pull-events-api.graybeach-3085945a.francecentral.azurecontainerapps.io (Swagger sur `/docs`, `/ask` protégé par l'en-tête `X-API-Key`).

## Les choix et leurs raisons

| Décision | Choix | Pourquoi |
|---|---|---|
| Région | France Central (Paris) | Données en France : argument RGPD pour des clients français |
| Base vectorielle | pgvector sur PostgreSQL Flexible Server | Gratuit 12 mois (B1ms) et même code qu'en local. Azure AI Search : le niveau Free (50 Mo) ne contient pas l'index (≈ 160 Mo de vecteurs en 1024 dimensions), le niveau Basic coûte ≈ 65 €/mois |
| LLM | `gpt-4.1-mini`, déploiement **Standard régional** | Modèle sans raisonnement, comme Ministral 8B. Le quota Data Zone de `gpt-4.1-mini` est à 0 sur un compte d'essai ; le régional garde en plus les requêtes en France |
| Embeddings | `text-embedding-3-small` (1536 dimensions), **Data Zone Standard** | Quota disponible ; traitement dans l'Union européenne |
| Mistral via Foundry | écarté | Foundry ne vend en direct que Mistral Large 3 et Medium 3.5 : ni Ministral 8B, ni `mistral-embed` |
| Image | GitHub Container Registry, public | Gratuit ; l'image ne contient aucun secret. Azure Container Registry Basic aurait coûté ≈ 4,4 €/mois |
| Secrets | Key Vault en mode RBAC, lus par l'identité managée de l'application | Aucun secret dans l'image ni dans la configuration de Container Apps, qui ne garde que des références |
| Observabilité | Langfuse Cloud, région UE | Offre gratuite ; Langfuse auto-hébergé demande ≈ 2 Go de RAM |
| Accès | Clé d'API sur `/ask`, `/rebuild` désactivé | URL publique : sans protection, n'importe qui consommerait des tokens ou lancerait une réindexation payante |

## Inventaire des ressources

Toutes dans l'abonnement « Azure subscription 1 » (« Plan Azure », paiement à l'utilisation, sans crédit ; 12 mois de services gratuits jusqu'au 24/10/2027), groupe de ressources **`rg-pull-events`**, région **`francecentral`**, étiquette `project=pull-events`. Le suffixe `cr24` rend uniques les noms qui doivent l'être dans tout Azure.

| Ressource | Nom | Réglages |
|---|---|---|
| Groupe de ressources | `rg-pull-events` | France Central |
| Budget | `budget-pull-events` (niveau abonnement) | 10 par mois dans la devise de facturation, du 01/10/2026 au 31/10/2027 ; e-mail aux propriétaires et à une adresse explicite à 50 %, 80 %, 100 % réels et 100 % prévus. Un budget alerte, il ne bloque rien |
| Key Vault | `kv-pull-events-cr24` | Standard, autorisation **RBAC**, suppression réversible 7 jours, protection contre la purge depuis l'audit du 25/10 |
| Foundry (Azure AI Services) | `ais-pull-events-cr24` | Type `AIServices`, SKU S0, sous-domaine `ais-pull-events-cr24` (point de terminaison `https://ais-pull-events-cr24.openai.azure.com/`) |
| Déploiement | `gpt-4.1-mini` | Modèle `gpt-4.1-mini` version `2025-04-14`, SKU **Standard**, capacité 50 à la création, **20** (20 000 tokens/minute) depuis l'audit du 25/10, filtre de contenu `Microsoft.DefaultV2` |
| Déploiement | `text-embedding-3-small` | Modèle `text-embedding-3-small` version `1`, SKU **DataZoneStandard**, capacité 350 à la création, **50** depuis l'audit du 25/10, filtre `Microsoft.DefaultV2` |
| PostgreSQL Flexible Server | `psql-pull-events-cr24` | PostgreSQL 17, Burstable **Standard_B1ms** (1 vCore, 2 Go), 32 Go de stockage, sauvegardes 7 jours, sans haute disponibilité, utilisateur `pgadmin`, accès public |
| Paramètre serveur | `azure.extensions` | `VECTOR` (pgvector doit être autorisé avant `CREATE EXTENSION`) |
| Règles de pare-feu | `allow-azure-services` (0.0.0.0) et `FirewallIPAddress_…` (IP du poste) | La première laisse entrer les services Azure, dont Container Apps : **risque résiduel**, voir [Sécurité](#sécurité). Terraform la supprime dès que le réseau privé est activé (`private_network = true`) |
| Base de données | `pullevents` | Extension `vector` 0.8.7 ; tables `event_chunks` (Azure OpenAI, 1536 dimensions) et `event_chunks_mistral` (copie de la base locale, 1024 dimensions), index HNSW sur chacune ; `CONNECT` retiré à `PUBLIC` |
| Rôle PostgreSQL | `pullevents_app` | Utilisé par l'API : `CONNECT` sur `pullevents`, `USAGE` sur `public`, **`SELECT` seulement** sur les tables, y compris celles recréées plus tard par `pgadmin` (`ALTER DEFAULT PRIVILEGES`) |
| Log Analytics | `log-pull-events` | Rétention 30 jours ; reçoit les journaux des conteneurs ; plafond d'ingestion 0,5 Go/jour depuis l'audit du 25/10 |
| Environnement Container Apps | `cae-pull-events` | Plan Consommation, journaux vers `log-pull-events` |
| Container App | `ca-pull-events-api` | Image `ghcr.io/clementrbl/pull-events:0.2.1-azure` à la création, `1.3.0` depuis le 25/10/2026 (géré par Terraform), 0,5 vCPU, 1 Go, **0 à 1 réplica** (mise à zéro sans trafic), ingress externe HTTPS vers le port 8000, mode de révision unique, identité managée système |
| Attribution de rôle | `Key Vault Secrets User` | Identité de `ca-pull-events-api`, sur chacun des cinq secrets qu'elle lit (sur tout le coffre à la création, restreint par l'audit du 25/10) |
| Attribution de rôle | `Key Vault Secrets Officer` | Ton compte sur `kv-pull-events-cr24` (pour écrire les secrets) |

### Secrets du Key Vault

| Secret | Contenu | Origine |
|---|---|---|
| `postgres-admin-password` | Mot de passe de `pgadmin` | Généré (`openssl rand`), jamais affiché |
| `postgres-app-password` | Mot de passe de `pullevents_app` | Généré, écrit avec `--file` |
| `database-url` | `postgresql+psycopg://pullevents_app:…@psql-pull-events-cr24.postgres.database.azure.com:5432/pullevents?sslmode=require` | Lu par l'API (rôle en lecture seule) |
| `database-url-admin` | Même URL avec `pgadmin` | Indexation et évaluation depuis le poste ; jamais donné à l'API |
| `api-key` | Clé exigée dans `X-API-Key` | Générée (`openssl rand -hex 32`), renouvelée le 25/10/2026 après l'audit |
| `azure-openai-api-key` | Clé 1 de `ais-pull-events-cr24` | `az cognitiveservices account keys list` |
| `langfuse-public-key`, `langfuse-secret-key` | Projet Langfuse Cloud `pull-events`, réservé à Azure | `.env` local |

La clé Mistral a servi à la colonne A de l'évaluation, puis a été supprimée et purgée du coffre : l'API n'en a plus besoin.

### Variables d'environnement de l'application

| Variable | Valeur |
|---|---|
| `LLM_PROVIDER` | `azure_openai` |
| `AZURE_OPENAI_ENDPOINT` | `https://ais-pull-events-cr24.openai.azure.com/` |
| `LLM_MODEL` / `EMBEDDING_MODEL` / `EMBEDDING_SIZE` | `gpt-4.1-mini` / `text-embedding-3-small` / `1536` (noms des déploiements) |
| `VECTOR_TABLE` | `event_chunks` |
| `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` | `0.484` / `1.936` (dollars, Standard régional France Central) |
| `EMBEDDING_PRICE_PER_MTOK` | `0.022` |
| `ENABLE_REBUILD` | `false` (c'est aussi la valeur par défaut) |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` |
| `DATABASE_URL`, `API_KEY`, `AZURE_OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `secretref:` vers le secret Key Vault du même nom (`database-url`, `api-key`, …) |

`LLM_MAX_TOKENS` (1024), `LLM_TIMEOUT_SECONDS` (60) et `LANGFUSE_TRACE_CONTENT` (true) gardent leur valeur par défaut. La démo trace le texte des questions ; passer `LANGFUSE_TRACE_CONTENT=false` pour ne garder que les mesures.

## Sécurité

Mesures appliquées après l'audit du 25/10/2026 :

| Risque | Mesure |
|---|---|
| URL publique : consommation de tokens par n'importe qui | Clé d'API (`X-API-Key`) + limites de débit : 10 questions/min par IP, 30/min par clé. Vérifié sur Azure : la 11ᵉ question de la minute reçoit 429 |
| Limite par IP contournable en forgeant `X-Forwarded-For` (audit 3.4) | L'API lit la **dernière** adresse de `X-Forwarded-For`, celle qu'ajoute l'ingress Envoy ; `--forwarded-allow-ips "*"` retiré du `Dockerfile`. Test `tests/test_api.py`. **Dans le code, pas encore dans l'image publiée** |
| Conteneur exécuté en root (audit 3.3) | Utilisateur `app` (uid 1000, le même que `runAsUser` du chart Helm) dans le `Dockerfile`. **Idem : image à republier** |
| Azure OpenAI appelé par clé (audit 4.1) | Code prêt pour l'identité managée (`DefaultAzureCredential`, sans `AZURE_OPENAI_API_KEY`). **Rôle et désactivation des clés à faire côté Azure**, voir [ci-dessous](#azure-openai-sans-clé--identité-managée-à-activer) |
| Réindexation déclenchée à distance | `/rebuild` désactivé par défaut, une exécution à la fois, une par jour |
| Réponse qui s'emballe ou bloque | `max_tokens=1024`, délai de 60 s sur le LLM et les embeddings |
| Fuite d'informations dans les erreurs | Messages génériques ; détail (dont `stderr`) dans les journaux Log Analytics |
| Compromission de l'API → écriture en base | Rôle `pullevents_app` en lecture seule ; `pgadmin` réservé au poste |
| Secrets dans l'historique du shell | `az keyvault secret set --file` sur un fichier temporaire en `umask 077`, supprimé après usage |
| Clés partagées entre environnements | Clé Mistral retirée du coffre ; projet Langfuse Cloud réservé à Azure (le local retrouve son Langfuse auto-hébergé) |
| Clé d'API vue en clair pendant les tests | Clé renouvelée |

**Risque résiduel accepté : la règle `allow-azure-services`.** Elle autorise le port 5432 depuis toute adresse Azure, y compris d'autres clients. Restent exigés : TLS, un mot de passe aléatoire de 35 caractères et, pour l'API, un rôle en lecture seule. En plan Consommation sans réseau virtuel, les adresses sortantes de Container Apps ne sont pas garanties stables (une liste d'adresses partagées, qui peut changer), donc impossible de les mettre seules en liste blanche de façon fiable. La vraie correction est un réseau virtuel (environnement Container Apps intégré au VNet, PostgreSQL joint par un point de terminaison privé) : écrite en Terraform (`network.tf`, bascule `private_network`), coût ≈ 7 €/mois, **non activée** le 25/10/2026 (aucun crédit Azure, règle zéro dépense ; détail et options chiffrées dans [terraform.md](terraform.md)). Pas fait non plus : restriction d'IP sur l'ingress (la démo doit rester joignable).

### Azure OpenAI sans clé : identité managée, à activer

Le code (`scripts/rag_chain.py`, `scripts/config.py`) n'exige plus `AZURE_OPENAI_API_KEY`. Sans elle, il obtient un jeton Entra ID avec `DefaultAzureCredential` (bibliothèque `azure-identity`) : l'identité managée de la Container App sur Azure, le compte `az login` sur le poste. Le jeton est renouvelé automatiquement ; le périmètre demandé est `https://cognitiveservices.azure.com/.default`. Avantage : plus aucun secret qui, s'il fuit, permette de consommer des tokens hors de l'API. Aucun coût : une attribution de rôle est gratuite.

Rien de ceci n'a été appliqué le 25/10/2026. Ordre à suivre, chaque étape vérifiable avant la suivante :

1. **Publier l'image** qui contient `azure-identity`, l'utilisateur non root et la nouvelle clé de limite de débit (par exemple `ghcr.io/clementrbl/pull-events:0.2.2-azure`), puis la référencer dans `infra/terraform/variables.tf` (variable `api_image`) et appliquer.
2. **Donner le rôle** `Cognitive Services OpenAI User` à l'identité managée de la Container App, sur le compte Foundry (le rôle autorise les appels d'inférence, pas la gestion du compte ni la lecture des clés) :

   ```bash
   az role assignment create \
     --assignee-object-id $(az containerapp show -g rg-pull-events -n ca-pull-events-api --query identity.principalId -o tsv) \
     --assignee-principal-type ServicePrincipal --role "Cognitive Services OpenAI User" \
     --scope $(az cognitiveservices account show -g rg-pull-events -n ais-pull-events-cr24 --query id -o tsv)
   ```

   Équivalent Terraform, prêt en commentaire dans `infra/terraform/ai.tf` (ressource `azurerm_role_assignment.openai_user_api`). La propagation d'un rôle prend quelques minutes.
3. **Retirer la variable** `AZURE_OPENAI_API_KEY` de la Container App : supprimer son bloc `env` dans `container_apps.tf` et `azure-openai-api-key` de `local.app_secret_names` dans `main.tf` (ou, à la main, `az containerapp update -g rg-pull-events -n ca-pull-events-api --remove-env-vars AZURE_OPENAI_API_KEY`). Vérifier qu'une question sur `/ask` répond 200 et que les journaux ne montrent pas de 401 venant d'Azure OpenAI.
4. **Couper les clés du compte** (`disable_local_auth`) : dans `ai.tf`, `local_auth_enabled = false` sur `azurerm_cognitive_account.ai`, ou `az resource update --ids <id du compte> --set properties.disableLocalAuth=true`. Dès lors, `key1`/`key2` ne servent plus à rien, même si elles fuient. Supprimer ensuite le secret `azure-openai-api-key` du Key Vault et l'attribution de rôle qui le lisait.

À savoir :

- Depuis le poste (indexation, évaluation), même chose : sans `AZURE_OPENAI_API_KEY`, `az login` suffit, à condition que ton compte ait aussi le rôle `Cognitive Services OpenAI User` sur le compte Foundry.
- Sur Kubernetes (`k8s/pull-events`), l'identité managée du pod demande *Workload Identity* (fédération entre un ServiceAccount et une identité Azure), que le chart ne configure pas : y garder la clé tant que l'étape 4 n'est pas faite, ou ajouter cette fédération d'abord.
- `AZURE_OPENAI_API_KEY` vide compte comme absente : pas besoin de la retirer de `docker-compose.yml`.

Renouveler la clé d'API :

```bash
umask 077; openssl rand -hex 32 | tr -d '\n' > /tmp/api-key
az keyvault secret set --vault-name kv-pull-events-cr24 -n api-key --file /tmp/api-key -o none && rm /tmp/api-key
az containerapp revision restart -g rg-pull-events -n ca-pull-events-api \
  --revision $(az containerapp show -g rg-pull-events -n ca-pull-events-api --query properties.latestRevisionName -o tsv)
```

## Coût

Prix de détail relevés le 24/10/2026 sur l'[API publique des prix Azure](https://prices.azure.com/api/retail/prices) (France Central).

| Ressource | Prix | Par mois (usage démo) | Avec les offres gratuites |
|---|---|---|---|
| Container Apps | au-delà de 180 000 vCPU-s, 360 000 Gio-s et 2 M requêtes gratuits par mois | ≈ 0 € (mise à zéro sans trafic) | 0 € |
| PostgreSQL B1ms | 0,0167 €/h + stockage | ≈ 12 € + ≈ 4 € | 0 € jusqu'au 24/10/2027 (750 h et 32 Go inclus, vérifié sur le portail le 25/10/2026) |
| Key Vault | 0,026 € / 10 000 opérations | < 0,01 € | |
| Log Analytics | facturé à l'ingestion au-delà de 5 Go/mois | ≈ 0 € | |
| `gpt-4.1-mini` Standard | 0,484 $ / 1,936 $ par million de tokens (entrée / sortie) | ≈ 1,3 $ pour 1 000 questions | en argent réel |
| `text-embedding-3-small` Data Zone | ≈ 0,022 $ par million de tokens (l'API des prix n'a pas de ligne Data Zone : 0,020 $ en Global, 0,022 à 0,027 $ en régional) | indexation complète ≈ 0,3 $, une fois | en argent réel |
| **Total** | | **≈ 16 €/mois** | **≈ 0 à 2 €/mois** |

D'après l'API Azure (audit du 25/10/2026), l'abonnement est au **paiement à l'utilisation** (`PayAsYouGo`, limite de dépense désactivée, carte active) et aucun crédit n'est visible ; les 12 mois de services gratuits ne sont pas vérifiables par l'API. Confirmé sur le portail le 25/10/2026 : aucun crédit, mais les **12 mois de services gratuits** (jusqu'au 24/10/2027) couvrent PostgreSQL Burstable 750 h/mois, 32 Go de stockage et 32 Go de sauvegarde : la base ne coûte rien pendant un an.

## Reproduire le déploiement

Prérequis : `az login`, et la CLI installée sans `sudo` avec :

```bash
uv tool install --python 3.12 --prerelease=allow "azure-cli==2.91.0"
```

Sans version imposée, uv installe la 2.0.67 (2019), qui ne démarre pas. `--prerelease=allow` est nécessaire parce que `azure-cli` 2.91 dépend d'une préversion de `azure-appconfiguration`.

Les commandes ci-dessous sont celles qui ont fonctionné, dans l'ordre (shell zsh).

```bash
RG=rg-pull-events; LOC=francecentral; KV=kv-pull-events-cr24
AI=ais-pull-events-cr24; PG=psql-pull-events-cr24; APP=ca-pull-events-api

# 0. Fournisseurs de ressources (une fois par abonnement)
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.DBforPostgreSQL Microsoft.KeyVault Microsoft.CognitiveServices; do
  az provider register -n $ns
done

# 1. Groupe de ressources et Key Vault (RBAC), avec le droit d'écrire des secrets
az group create -n $RG -l $LOC --tags project=pull-events
az keyvault create -n $KV -g $RG -l $LOC --enable-rbac-authorization true --retention-days 7
az role assignment create --assignee-object-id $(az ad signed-in-user show --query id -o tsv) \
  --assignee-principal-type User --role "Key Vault Secrets Officer" \
  --scope $(az keyvault show -n $KV --query id -o tsv)

# 2. Secrets générés directement dans le coffre
# --file plutôt que --value : la valeur n'apparaît ni dans l'historique ni dans la liste des processus
umask 077
printf '%s' "Aa1$(openssl rand -base64 24 | tr -d '/+=')" > /tmp/pg-pwd
az keyvault secret set --vault-name $KV -n postgres-admin-password --file /tmp/pg-pwd -o none && rm /tmp/pg-pwd
openssl rand -hex 32 | tr -d '\n' > /tmp/api-key
az keyvault secret set --vault-name $KV -n api-key --file /tmp/api-key -o none && rm /tmp/api-key

# 3. Budget : PUT REST sur Microsoft.Consumption/budgets (voir le corps JSON plus haut)
az rest --method put \
  --url "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.Consumption/budgets/budget-pull-events?api-version=2023-05-01" \
  --body @budget.json

# 4. Foundry et déploiements de modèles (vérifier d'abord les quotas)
az cognitiveservices usage list -l $LOC -o table
az cognitiveservices account create -n $AI -g $RG -l $LOC --kind AIServices --sku S0 --custom-domain $AI --yes
az cognitiveservices account deployment create -n $AI -g $RG --deployment-name gpt-4.1-mini \
  --model-name gpt-4.1-mini --model-version 2025-04-14 --model-format OpenAI --sku-name Standard --sku-capacity 50
az cognitiveservices account deployment create -n $AI -g $RG --deployment-name text-embedding-3-small \
  --model-name text-embedding-3-small --model-version 1 --model-format OpenAI --sku-name DataZoneStandard --sku-capacity 350
az keyvault secret set --vault-name $KV -n azure-openai-api-key \
  --value "$(az cognitiveservices account keys list -n $AI -g $RG --query key1 -o tsv)" -o none

# 5. PostgreSQL + pgvector
az postgres flexible-server create -g $RG -n $PG -l $LOC --tier Burstable --sku-name Standard_B1ms \
  --storage-size 32 --version 17 --admin-user pgadmin \
  --admin-password "$(az keyvault secret show --vault-name $KV -n postgres-admin-password --query value -o tsv)" \
  --public-access $(curl -s https://api.ipify.org) --zonal-resiliency Disabled --backup-retention 7 --yes
az postgres flexible-server db create -g $RG -s $PG -n pullevents
az postgres flexible-server firewall-rule create -g $RG -s $PG -n allow-azure-services \
  --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
az postgres flexible-server parameter set -g $RG -s $PG --name azure.extensions --value VECTOR
# puis, en SQL dans la base pullevents : CREATE EXTENSION vector;

# 6. Environnement Container Apps et journaux
az monitor log-analytics workspace create -g $RG -n log-pull-events -l $LOC --retention-time 30
az containerapp env create -g $RG -n cae-pull-events -l $LOC \
  --logs-workspace-id $(az monitor log-analytics workspace show -g $RG -n log-pull-events --query customerId -o tsv) \
  --logs-workspace-key $(az monitor log-analytics workspace get-shared-keys -g $RG -n log-pull-events --query primarySharedKey -o tsv)

# 7. Application, identité managée, accès au coffre
az containerapp create -g $RG -n $APP --environment cae-pull-events \
  --image ghcr.io/clementrbl/pull-events:0.2.1-azure --system-assigned \
  --ingress external --target-port 8000 --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi
az role assignment create --assignee-object-id $(az containerapp show -g $RG -n $APP --query identity.principalId -o tsv) \
  --assignee-principal-type ServicePrincipal --role "Key Vault Secrets User" \
  --scope $(az keyvault show -n $KV --query id -o tsv)

# 8. Rôle PostgreSQL de l'application (lecture seule), en SQL connecté en pgadmin :
#   CREATE ROLE pullevents_app LOGIN PASSWORD '…';
#   REVOKE ALL ON DATABASE pullevents FROM PUBLIC;
#   GRANT CONNECT ON DATABASE pullevents TO pullevents_app;
#   GRANT USAGE ON SCHEMA public TO pullevents_app;
#   GRANT SELECT ON ALL TABLES IN SCHEMA public TO pullevents_app;
#   ALTER DEFAULT PRIVILEGES FOR ROLE pgadmin IN SCHEMA public GRANT SELECT ON TABLES TO pullevents_app;
# Le mot de passe et les deux URL (database-url → pullevents_app, database-url-admin → pgadmin)
# vont dans le coffre avec --file, jamais en argument.

# 9. Secrets en référence Key Vault, puis variables d'environnement
KVURI=https://$KV.vault.azure.net/secrets
S=(); for s in database-url api-key azure-openai-api-key langfuse-public-key langfuse-secret-key; do
  S+=("$s=keyvaultref:$KVURI/$s,identityref:system"); done
az containerapp secret set -g $RG -n $APP --secrets "${S[@]}"
az containerapp update -g $RG -n $APP --set-env-vars \
  LLM_PROVIDER=azure_openai AZURE_OPENAI_ENDPOINT=https://$AI.openai.azure.com/ \
  LLM_MODEL=gpt-4.1-mini EMBEDDING_MODEL=text-embedding-3-small EMBEDDING_SIZE=1536 VECTOR_TABLE=event_chunks \
  LLM_INPUT_PRICE_PER_MTOK=0.484 LLM_OUTPUT_PRICE_PER_MTOK=1.936 EMBEDDING_PRICE_PER_MTOK=0.022 \
  ENABLE_REBUILD=false LANGFUSE_BASE_URL=https://cloud.langfuse.com \
  DATABASE_URL=secretref:database-url API_KEY=secretref:api-key AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key \
  LANGFUSE_PUBLIC_KEY=secretref:langfuse-public-key LANGFUSE_SECRET_KEY=secretref:langfuse-secret-key
```

### Remplir la base

L'API ne reconstruit pas l'index dans le cloud (`ENABLE_REBUILD=false`) : on indexe depuis le poste, sur la base Azure.

```bash
# Colonne A : copie des vecteurs Mistral de la base locale (aucun appel d'API)
docker exec pull-events-db-1 pg_dump -U pullevents -d pullevents -t event_chunks --no-owner --no-privileges \
  | docker run -i --rm -e PGPASSWORD pgvector/pgvector:pg17 psql "host=$PG.postgres.database.azure.com user=pgadmin dbname=pullevents sslmode=require"
# puis renommer la table et ses index en event_chunks_mistral*

# Colonne B : réindexation avec Azure OpenAI
LLM_PROVIDER=azure_openai EMBEDDING_MODEL=text-embedding-3-small EMBEDDING_SIZE=1536 \
AZURE_OPENAI_ENDPOINT=https://$AI.openai.azure.com/ \
AZURE_OPENAI_API_KEY=$(az keyvault secret show --vault-name $KV -n azure-openai-api-key --query value -o tsv) \
DATABASE_URL=$(az keyvault secret show --vault-name $KV -n database-url-admin --query value -o tsv) \
uv run python -m scripts.build_index
```

### Appeler l'API

```bash
KEY=$(az keyvault secret show --vault-name kv-pull-events-cr24 -n api-key --query value -o tsv)
curl -X POST https://ca-pull-events-api.graybeach-3085945a.francecentral.azurecontainerapps.io/ask \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"question": "Y a-t-il des expositions à Versailles ?"}'
```

Le premier appel après une période sans trafic prend quelques secondes de plus : le réplica redémarre (mise à zéro).

## Pièges rencontrés

| Symptôme | Cause | Correction |
|---|---|---|
| `az` : `No module named 'pkg_resources'` | uv avait installé `azure-cli` 2.0.67 | Imposer `azure-cli==2.91.0` et Python 3.12 |
| `unrecognized arguments: --high-availability` | Option renommée | `--zonal-resiliency Disabled` |
| `--database-name ... only applies to elastic clusters` | Option réservée aux clusters élastiques | Créer la base avec `az postgres flexible-server db create` |
| `Secret name has an invalid value ' database-url'` | zsh ne découpe pas une variable non entre guillemets | Construire la liste dans un tableau (`"${S[@]}"`) |
| Quota Data Zone de `gpt-4.1-mini` à 0 | Limite des abonnements d'essai | Déploiement Standard régional |
| `pass not initialized` au `docker login` | Gestionnaire d'identifiants Docker non configuré | `DOCKER_CONFIG` temporaire, supprimé après le push |
| Visibilité GHCR inchangeable par l'API | L'API REST GitHub ne le permet pas | *Package settings → Change visibility* |
| Juste après `az containerapp update`, l'API répond encore avec l'ancienne configuration | La nouvelle révision s'active en une à deux minutes | Vérifier avec `az containerapp revision list` |

## Détruire

Tout supprimer (irréversible) :

```bash
az group delete -n rg-pull-events --yes
# Key Vault et Foundry restent en suppression réversible : les purger pour libérer les noms
az keyvault purge -n kv-pull-events-cr24 -l francecentral
az cognitiveservices account purge -n ais-pull-events-cr24 -g rg-pull-events -l francecentral
# Le budget est au niveau de l'abonnement, hors du groupe
az rest --method delete --url "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.Consumption/budgets/budget-pull-events?api-version=2023-05-01"
```

Pour seulement arrêter les frais de PostgreSQL sans rien perdre : `az postgres flexible-server stop -g rg-pull-events -n psql-pull-events-cr24`. Azure le redémarre automatiquement au bout de 7 jours.

L'image publique reste sur GHCR : la supprimer depuis les paramètres du paquet sur GitHub si besoin.
