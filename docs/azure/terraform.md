# Infrastructure as code : Terraform

Le déploiement Azure fait à la main le 24 octobre 2026 ([deploiement.md](deploiement.md)) est maintenant décrit dans `infra/terraform/`. Aucune ressource n'a été recréée : chaque ressource existante a été **importée** dans l'état Terraform, puis `terraform plan` est resté vide, ce qui prouve que le code décrit exactement la réalité.

```
$ terraform plan
No changes. Your infrastructure matches the configuration.
```

## Ce que ça apporte

- **Une seule source de vérité.** Quotas, tailles, règles de pare-feu, droits : tout est lisible dans quelques fichiers `.tf`, relu en revue de code, versionné.
- **Un écart devient visible.** Une modification faite à la main dans le portail apparaît au prochain `plan` comme une différence à corriger, dans un sens ou dans l'autre.
- **Reproductible et destructible.** Tout recréer ailleurs, ou tout détruire pour arrêter la facturation, tient en une commande.

## Fichiers

| Fichier | Contenu |
|---|---|
| `versions.tf` | Terraform ≥ 1.17, provider `azurerm` 5.x |
| `backend.tf` | Où vit l'état (voir [État et verrou](#état-et-verrou)) |
| `providers.tf` | Abonnement, options du provider |
| `variables.tf` | Abonnement, région, suffixe, IP du poste, image, budget, bascules `private_network` et `enable_aks` |
| `main.tf` | Groupe de ressources, Key Vault, droits sur les secrets |
| `ai.tf` | Foundry (Azure AI Services) et ses deux déploiements de modèles |
| `postgres.tf` | PostgreSQL Flexible Server, base, paramètre `azure.extensions`, règles de pare-feu |
| `container_apps.tf` | Log Analytics, environnement Container Apps, application API |
| `budget.tf` | Budget mensuel et ses quatre alertes |
| `network.tf` | Phase réseau privé : VNet, sous-réseaux, zone DNS privée, Private Endpoint (`private_network = true`) |
| `aks.tf` | Phase Kubernetes : cluster AKS, identité, IP publique (`enable_aks = true`) |
| `imports.tf` | Blocs `import` : d'où vient chaque ressource |
| `outputs.tf` | URL de l'API, noms, identifiants utiles au chart Helm |
| `terraform.tfvars.example` | Modèle des valeurs propres au poste (`terraform.tfvars` est ignoré par git) |
| `bootstrap-backend.sh` | Crée une fois le compte de stockage de l'état |

## Importer plutôt que recréer

Les blocs `import` de `imports.tf` sont déclaratifs (Terraform ≥ 1.5) : au premier `apply`, Terraform adopte chaque ressource dans son état au lieu de la créer. Ils restent dans le code comme trace d'origine et sont ignorés ensuite.

Le premier `plan` a montré trois écarts entre le code et la réalité, tous corrigés dans le code et non dans Azure :

| Écart | Correction |
|---|---|
| Foundry avait `project_management_enabled = true` (posé par la CLI) | Valeur reprise dans `ai.tf` : la changer forcerait une recréation |
| L'attribut `endpoint` de Foundry vaut `…cognitiveservices.azure.com`, l'application utilise `…openai.azure.com` | Point de terminaison écrit à partir du sous-domaine |
| Container Apps renvoie `identity = "system"` pour chaque secret, le provider exige `"System"` | `lifecycle { ignore_changes = [secret] }` sur l'application, avec commentaire : c'est un défaut du provider, pas de l'infra |

Un seul changement a été appliqué pendant l'import : Log Analytics s'est vu écrire explicitement `local_authentication_enabled = true`, sa valeur effective.

## Secrets

Règle : **aucune valeur de secret dans le code, et le moins possible dans l'état**.

- Les secrets vivent dans le Key Vault, écrits avec `az keyvault secret set --file` (jamais en argument de commande). Terraform ne les gère pas : il ne connaît que leurs **URI**, que Container Apps lit avec son identité managée.
- Le mot de passe administrateur PostgreSQL n'est pas déclaré : Terraform n'en a pas besoin pour gérer le serveur, il resterait sinon dans l'état.
- Inévitable : le provider exporte dans l'état les clés d'accès de Foundry et de Log Analytics (attributs `primary_access_key`, `primary_shared_key`). D'où un état **chiffré au repos, accessible par identité Entra ID seulement, versionné** (section suivante), et jamais versionné dans git (`.gitignore`).
- Valeurs propres au poste (identifiant d'abonnement, IP publique, e-mail) : `terraform.tfvars`, ignoré par git, avec `terraform.tfvars.example` comme modèle.

## État et verrou

Cible : un compte de stockage Azure dédié, créé une fois par `bootstrap-backend.sh` (Terraform ne peut pas créer son propre backend) :

- groupe `rg-pull-events-tfstate`, compte `stpulleventstfcr24`, conteneur `tfstate` ;
- clés de compte désactivées, accès par identité Entra ID (`use_azuread_auth = true`, rôle *Storage Blob Data Contributor*) ;
- versioning des blobs : chaque écriture de l'état garde la version précédente ;
- **verrou** : Terraform prend un bail (*lease*) sur le blob pendant chaque `plan`/`apply`. Deux exécutions simultanées sont impossibles, c'est ce qui évite deux personnes (ou un poste et une CI) qui écrivent l'état en même temps ;
- coût : quelques centimes par mois.

Décision du 25/10/2026 : pas de dépense, même de quelques centimes, tant que le projet est une démonstration. `backend.tf` garde donc un état **local** (`terraform.tfstate`, ignoré par git, un seul poste). Le jour où le projet est partagé ou déployé par une CI, passer à l'état distant :

```bash
cd infra/terraform
./bootstrap-backend.sh
# remplacer backend "local" {} par le bloc azurerm commenté dans backend.tf, puis :
terraform init -migrate-state
```

## Appliquer

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # puis remplir
terraform init
terraform plan -out=plan.tfplan     # relire le plan : rien ne doit être recréé par surprise
terraform apply plan.tfplan
```

Toujours `plan` avant `apply`, et appliquer le fichier de plan relu : `apply` sans fichier recalcule un plan qui peut différer.

Deux bascules dans `terraform.tfvars` :

| Variable | Effet | Coût |
|---|---|---|
| `private_network = true` | VNet, zone DNS privée, Private Endpoint PostgreSQL, environnement Container Apps recréé dans le VNet (nouvelle URL), règle de pare-feu `0.0.0.0` supprimée | ≈ 7 €/mois |
| `enable_aks = true` | Cluster AKS d'un nœud dans le VNet (demande `private_network = true`) | ≈ 0,07 €/h |

## Durcissement appliqué après l'audit du 25/10/2026

| Constat de l'audit | Changement Terraform |
|---|---|
| 4.2 Quotas TPM trop hauts (seule borne dure de facturation si une clé fuit) | `gpt-4.1-mini` 50 K → **20 K** tokens/min, embeddings 350 K → **50 K** |
| 7.2 Log Analytics sans plafond | `daily_quota_gb = 0.5` |
| 6.1 Budget sans adresse explicite | `contact_emails` sur les quatre alertes |
| 5.2 L'application lisait tout le coffre, y compris les secrets administrateur | Rôle *Key Vault Secrets User* attribué **secret par secret** (5 attributions) |
| 5.3 Pas de protection contre la purge du coffre | `purge_protection_enabled = true` (irréversible) |
| 2.1 Règle `0.0.0.0` « services Azure » | **Risque accepté** (voir ci-dessous) ; supprimée dès que `private_network = true` |

### La règle `0.0.0.0`, décision du 25/10/2026

Trois options ont été chiffrées :

| Option | Coût | Verdict |
|---|---|---|
| Réseau privé : VNet, Private Endpoint, environnement Container Apps recréé dans le VNet | ≈ 7 €/mois | La vraie correction. Code écrit (`network.tf`), planifié, **non appliqué** : l'abonnement n'a aucun crédit, tout serait en argent réel |
| Remplacer `0.0.0.0` par les adresses sortantes de l'environnement Container Apps | 0 € | Le plan a révélé **181 adresses**, le pool partagé de Container Apps en France Central : on passerait de « tout Azure » à « tout client Container Apps de la région », pour 181 règles qui peuvent changer sans prévenir. Gain trop faible pour le prix en lisibilité |
| Garder la règle, documenter le risque | 0 € | **Retenu.** Remparts : TLS imposé, mot de passe aléatoire de 35 caractères, rôle `pullevents_app` en lecture seule pour l'API, quotas TPM réduits |

Restent à faire côté code : appel d'Azure OpenAI par identité managée (`disable_local_auth`), dernière adresse de `X-Forwarded-For` pour la limite de débit, utilisateur non root dans l'image (fait sur Kubernetes par `runAsNonRoot`).

## Coût

Prix de détail France Central relevés le 25/10/2026 sur l'API publique des prix.

| Brique | Prix | Par mois |
|---|---|---|
| État distant (Storage LRS) | opérations + quelques Ko | < 0,05 € |
| VNet, sous-réseaux, environnement Container Apps Consumption | gratuit | 0 € |
| Zone DNS privée | 0,44 €/zone | 0,44 € |
| Private Endpoint | 0,0088 €/h + 0,0088 €/Go | ≈ 6,4 € |
| AKS plan de contrôle Free | gratuit | 0 € |
| AKS nœud `Standard_B2als_v2` (2 vCPU, 4 Go) | 0,037 €/h | ≈ 27 € |
| AKS Load Balancer Standard | 0,022 €/h | ≈ 16 € |
| AKS IP publique Standard | 0,0044 €/h | ≈ 3,2 € |
| AKS disque OS 32 Go (E4) | 2,3 €/mois | 2,3 € |
| **AKS complet** | **≈ 0,067 €/h** | **≈ 48 €, ou 1,6 € par jour de démo** |

Vérifié sur le portail le 25/10/2026 : abonnement **« Plan Azure »** (paiement à l'utilisation), **aucun crédit**, mais les **12 mois de services gratuits** sont actifs jusqu'au 24/10/2027 et couvrent PostgreSQL Flexible Server Burstable (750 h/mois), 32 Go de stockage et 32 Go de sauvegarde : la base ne coûte rien pendant un an. Tout le reste serait en argent réel, d'où la décision de ne pas activer `private_network` ni `enable_aks`.

### AKS : prouvé par un plan, pas par un apply

`terraform plan -var enable_aks=true -var private_network=true` (lecture seule, 25/10/2026) :

```
Plan: 23 to add, 0 to change, 8 to destroy.
```

À créer : VNet et trois sous-réseaux, zone DNS privée et son lien, Private Endpoint PostgreSQL, identité du cluster et son rôle *Network Contributor*, IP publique avec étiquette DNS, cluster AKS, cinq rôles Key Vault pour l'add-on CSI. À remplacer : l'environnement Container Apps et l'application (le VNet ne s'ajoute pas après coup), et les cinq rôles de l'application (leur identité change avec elle). À détruire : la règle `0.0.0.0`. Rien d'inattendu : le code est valide, l'application reviendrait à activer deux variables.

## Détruire

```bash
# AKS seulement (arrête la facturation horaire, garde le reste)
terraform apply -var enable_aks=false

# Tout
terraform destroy
az keyvault purge -n kv-pull-events-cr24 -l francecentral       # impossible avant 7 jours : purge protection
az cognitiveservices account purge -n ais-pull-events-cr24 -g rg-pull-events -l francecentral
```

## Pièges rencontrés

| Symptôme | Cause | Correction |
|---|---|---|
| `SubscriptionNotFound` sur `az storage account check-name` | Fournisseur `Microsoft.Storage` non enregistré | `az provider register -n Microsoft.Storage` (idem `Network`, `ContainerService`, `Compute`) |
| Plan jamais vide : les cinq secrets de la Container App réécrits à chaque fois | Provider : `identity` attendu `"System"`, Azure renvoie `"system"` | `ignore_changes = [secret]` |
| `project_management_enabled` force la recréation de Foundry | La CLI active les projets Foundry à la création | Reprendre `true` dans le code |
| `az postgres flexible-server firewall-rule list` : `unrecognized arguments: -n` | Le serveur se passe avec `-s` / `--name` selon la commande | `-s psql-pull-events-cr24` |
