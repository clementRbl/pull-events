# Foundry (Azure AI Services) et ses deux déploiements de modèles.

resource "azurerm_cognitive_account" "ai" {
  name                  = "ais-pull-events-${var.name_suffix}"
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  kind                  = "AIServices"
  sku_name              = "S0"
  custom_subdomain_name = "ais-pull-events-${var.name_suffix}"
  # Activé par la CLI à la création (projets Foundry) ; irréversible sans recréer le compte.
  project_management_enabled = true
  tags                       = local.tags

  identity {
    type = "SystemAssigned"
  }
}

# Standard régional : les requêtes restent en France. Capacité en milliers de
# tokens par minute : la seule borne dure de facturation si une clé fuit
# (audit 4.2). 20 000 tokens/min couvrent 10 questions/min à ≈ 1 400 tokens.
resource "azurerm_cognitive_deployment" "gpt" {
  name                 = "gpt-4.1-mini"
  cognitive_account_id = azurerm_cognitive_account.ai.id
  rai_policy_name      = "Microsoft.DefaultV2"

  model {
    format  = "OpenAI"
    name    = "gpt-4.1-mini"
    version = "2025-04-14"
  }

  sku {
    name     = "Standard"
    capacity = 20
  }
}

# Data Zone : traitement dans l'Union européenne (le quota régional était à 0).
# 50 000 tokens/min : une question ≈ 100 tokens ; l'indexation complète se fait
# depuis le poste et peut attendre.
resource "azurerm_cognitive_deployment" "embeddings" {
  name                 = "text-embedding-3-small"
  cognitive_account_id = azurerm_cognitive_account.ai.id
  rai_policy_name      = "Microsoft.DefaultV2"

  model {
    format  = "OpenAI"
    name    = "text-embedding-3-small"
    version = "1"
  }

  sku {
    name     = "DataZoneStandard"
    capacity = 50
  }
}
