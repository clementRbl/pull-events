# Groupe de ressources, Key Vault et droits d'accès au coffre.

data "azurerm_client_config" "current" {}

locals {
  tags = {
    project = "pull-events"
  }
}

resource "azurerm_resource_group" "main" {
  name     = "rg-pull-events"
  location = var.location
  tags     = local.tags
}

# Les valeurs des secrets ne sont pas gérées par Terraform : elles sont écrites
# dans le coffre avec `az keyvault secret set --file` (voir docs/azure/deploiement.md)
# et n'apparaissent donc jamais dans l'état.
resource "azurerm_key_vault" "main" {
  name                       = "kv-pull-events-${var.name_suffix}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  rbac_authorization_enabled = true
  soft_delete_retention_days = 7
  # Audit 5.3 : un coffre protégé ne peut pas être purgé avant la fin de la
  # rétention (7 jours). Irréversible une fois activé.
  purge_protection_enabled = true
  # Créé sans étiquette par la CLI ; laissé tel quel pour que le plan reste vide.
}

# Secrets que l'application lit (références Container Apps, pilote CSI sur AKS).
# Les secrets d'administration (database-url-admin, postgres-admin-password)
# n'en font pas partie.
locals {
  app_secret_names = toset([
    "api-key",
    "azure-openai-api-key",
    "database-url",
    "langfuse-public-key",
    "langfuse-secret-key",
  ])
}

# Ton compte écrit les secrets.
resource "azurerm_role_assignment" "kv_secrets_officer_me" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
  principal_type       = "User"
}

# L'identité managée de l'API lit ses secrets, et seulement ceux-là :
# le rôle est attribué secret par secret (audit 5.2), pas sur tout le coffre.
resource "azurerm_role_assignment" "kv_secrets_user_api" {
  for_each             = local.app_secret_names
  scope                = "${azurerm_key_vault.main.id}/secrets/${each.value}"
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_container_app.api.identity[0].principal_id
  principal_type       = "ServicePrincipal"
}
