# Réseau privé (phase 2), activé par var.private_network.
#
# Avant : PostgreSQL public, règle « services Azure » 0.0.0.0 (tout client Azure
# peut joindre le port 5432). Après : les conteneurs passent par le VNet et un
# point de terminaison privé ; l'accès public ne garde que l'IP du poste.
#
# Coût (France Central, octobre 2026) : Private Endpoint 0,0088 €/h ≈ 6,4 €/mois,
# zone DNS privée 0,44 €/mois ; VNet, sous-réseaux et environnement Container
# Apps (profil Consumption) : 0 €.

resource "azurerm_virtual_network" "main" {
  count               = var.private_network ? 1 : 0
  name                = "vnet-pull-events"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.10.0.0/16"]
  tags                = local.tags
}

# Container Apps (environnement « workload profiles ») : /27 minimum, délégué.
resource "azurerm_subnet" "aca" {
  count                = var.private_network ? 1 : 0
  name                 = "snet-aca"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main[0].name
  address_prefixes     = ["10.10.0.0/27"]

  delegation {
    name = "container-apps"
    service_delegation {
      name    = "Microsoft.App/environments"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

# Points de terminaison privés.
resource "azurerm_subnet" "private_endpoints" {
  count                = var.private_network ? 1 : 0
  name                 = "snet-pe"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main[0].name
  address_prefixes     = ["10.10.1.0/28"]
}

# Nœuds AKS.
resource "azurerm_subnet" "aks" {
  count                = var.private_network ? 1 : 0
  name                 = "snet-aks"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main[0].name
  address_prefixes     = ["10.10.2.0/24"]
}

# Dans le VNet, le nom public du serveur se résout vers l'IP privée du point
# de terminaison grâce à cette zone ; hors du VNet, il garde son IP publique.
resource "azurerm_private_dns_zone" "postgres" {
  count               = var.private_network ? 1 : 0
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  count               = var.private_network ? 1 : 0
  name                = "vnet-pull-events"
  private_dns_zone_id = azurerm_private_dns_zone.postgres[0].id
  virtual_network_id  = azurerm_virtual_network.main[0].id
  tags                = local.tags
}

resource "azurerm_private_endpoint" "postgres" {
  count               = var.private_network ? 1 : 0
  name                = "pe-psql-pull-events"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints[0].id
  tags                = local.tags

  private_service_connection {
    name                           = "psql-pull-events"
    private_connection_resource_id = azurerm_postgresql_flexible_server.main.id
    subresource_names              = ["postgresqlServer"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "postgres"
    private_dns_zone_ids = [azurerm_private_dns_zone.postgres[0].id]
  }
}
