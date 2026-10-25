# PostgreSQL Flexible Server + pgvector.

resource "azurerm_postgresql_flexible_server" "main" {
  name                = "psql-pull-events-${var.name_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags

  version  = "17"
  sku_name = "B_Standard_B1ms"
  zone     = "2"

  storage_mb        = 32768
  storage_tier      = "P4"
  auto_grow_enabled = false

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false

  administrator_login = "pgadmin"
  # Le mot de passe a été posé à la création (CLI) et vit dans le Key Vault
  # (secret postgres-admin-password). Terraform n'en a pas besoin pour gérer
  # le serveur : ne pas l'ajouter ici, il finirait dans l'état.

  authentication {
    password_auth_enabled         = true
    active_directory_auth_enabled = false
  }

  public_network_access_enabled = true
}

resource "azurerm_postgresql_flexible_server_database" "pullevents" {
  name      = "pullevents"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# pgvector doit être autorisé côté serveur avant `CREATE EXTENSION vector`.
resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "VECTOR"
}

# Seule IP autorisée en accès public : le poste de travail (indexation, évaluation).
resource "azurerm_postgresql_flexible_server_firewall_rule" "workstation" {
  name             = "FirewallIPAddress_2026-10-24_23-18-40"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = var.workstation_ip
  end_ip_address   = var.workstation_ip
}

# Règle spéciale « services Azure » (0.0.0.0) : tout client hébergé dans Azure
# peut joindre le port 5432. Risque accepté le 25/10/2026 (audit 2.1), pas de
# dépense : l'environnement Container Apps sort par un pool de 181 adresses
# partagées, les lister serait à peine plus étroit ; la vraie correction est
# le réseau privé (var.private_network, ≈ 7 €/mois), qui supprime cette règle.
# Remparts restants : TLS, mot de passe de 35 caractères, rôle en lecture seule.
resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure_services" {
  count            = var.private_network ? 0 : 1
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}
