# Cluster AKS (phase 3), activé par var.enable_aks. Demande var.private_network :
# les nœuds vont dans snet-aks et joignent PostgreSQL par le point de terminaison privé.
#
# Coût (France Central, octobre 2026) : plan de contrôle Free 0 €, un nœud
# Standard_B2als_v2 (2 vCPU, 4 Go) 0,037 €/h, Load Balancer Standard 0,022 €/h,
# IP publique 0,0044 €/h, disque OS 32 Go ≈ 0,003 €/h : ≈ 0,067 €/h, 1,6 €/jour.

# Identité du cluster : doit pouvoir attacher les nœuds au sous-réseau et
# utiliser l'IP publique du groupe de ressources (rôle Network Contributor).
resource "azurerm_user_assigned_identity" "aks" {
  count               = var.enable_aks ? 1 : 0
  name                = "id-aks-pull-events"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_role_assignment" "aks_network" {
  count                = var.enable_aks ? 1 : 0
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_user_assigned_identity.aks[0].principal_id
  principal_type       = "ServicePrincipal"
}

# IP publique de l'ingress, avec un nom DNS gratuit pour le certificat Let's Encrypt :
# pull-events-<suffixe>.francecentral.cloudapp.azure.com
resource "azurerm_public_ip" "ingress" {
  count               = var.enable_aks ? 1 : 0
  name                = "pip-aks-ingress"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"
  domain_name_label   = "pull-events-${var.name_suffix}"
  tags                = local.tags
}

resource "azurerm_kubernetes_cluster" "main" {
  count               = var.enable_aks ? 1 : 0
  name                = "aks-pull-events"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "pull-events-${var.name_suffix}"
  sku_tier            = "Free"
  tags                = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aks[0].id]
  }

  default_node_pool {
    name            = "system"
    vm_size         = "Standard_B2als_v2"
    node_count      = 1
    os_disk_size_gb = 32
    vnet_subnet_id  = azurerm_subnet.aks[0].id

    upgrade_settings {
      max_surge = "10%"
    }
  }

  node_provisioning_profile {
    mode = "Manual"
  }

  # Azure CNI en mode overlay : les pods ont leurs propres adresses (pod_cidr),
  # le sous-réseau ne porte que les nœuds. network_policy = azure applique les
  # NetworkPolicy du chart.
  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    network_policy      = "azure"
    pod_cidr            = "10.244.0.0/16"
    service_cidr        = "10.0.0.0/16"
    dns_service_ip      = "10.0.0.10"
    load_balancer_sku   = "standard"
    outbound_type       = "loadBalancer"
  }

  # Add-on Secrets Store CSI : les pods lisent les secrets du Key Vault.
  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  depends_on = [azurerm_role_assignment.aks_network]
}

# L'identité de l'add-on CSI lit les secrets de l'application.
resource "azurerm_role_assignment" "aks_kv_secrets" {
  for_each             = var.enable_aks ? local.app_secret_names : toset([])
  scope                = "${azurerm_key_vault.main.id}/secrets/${each.value}"
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_kubernetes_cluster.main[0].key_vault_secrets_provider[0].secret_identity[0].object_id
  principal_type       = "ServicePrincipal"
}
