#!/usr/bin/env bash
# Crée une fois le compte de stockage qui garde l'état Terraform.
# Terraform ne peut pas créer son propre backend : c'est la seule ressource
# créée hors Terraform après cette étape.
set -euo pipefail

RG=rg-pull-events-tfstate
SA=stpulleventstfcr24
LOC=francecentral

az group create -n "$RG" -l "$LOC" --tags project=pull-events -o none

# Clés de compte désactivées : accès par identité Entra ID seulement (az login).
# Versioning : chaque écriture de l'état garde la version précédente.
az storage account create -n "$SA" -g "$RG" -l "$LOC" --sku Standard_LRS --kind StorageV2 \
  --min-tls-version TLS1_2 --allow-blob-public-access false --allow-shared-key-access false \
  --tags project=pull-events -o none
az storage account blob-service-properties update --account-name "$SA" -g "$RG" \
  --enable-versioning true --enable-delete-retention true --delete-retention-days 7 -o none

# Le rôle Owner ne donne pas accès aux données des blobs : il faut un rôle « data ».
az role assignment create --role "Storage Blob Data Contributor" \
  --assignee-object-id "$(az ad signed-in-user show --query id -o tsv)" --assignee-principal-type User \
  --scope "$(az storage account show -n "$SA" -g "$RG" --query id -o tsv)" -o none

# Le rôle met parfois une minute à se propager.
sleep 30
az storage container create -n tfstate --account-name "$SA" --auth-mode login -o none
echo "Backend prêt : $SA/tfstate"
