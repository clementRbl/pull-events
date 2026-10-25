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

# Audit 4.1 : appeler Azure OpenAI par identité managée plutôt que par clé.
# Proposition, NON appliquée : le code de l'API sait déjà se passer de
# AZURE_OPENAI_API_KEY, mais l'image publiée ne le contient pas encore.
# Marche à suivre : docs/azure/deploiement.md, section « Azure OpenAI sans clé ».
#
# 1. Le rôle (gratuit) pour l'identité de la Container App :
#
# resource "azurerm_role_assignment" "openai_user_api" {
#   scope                = azurerm_cognitive_account.ai.id
#   role_definition_name = "Cognitive Services OpenAI User"
#   principal_id         = azurerm_container_app.api.identity[0].principal_id
#   principal_type       = "ServicePrincipal"
# }
#
# 2. Retirer le bloc env AZURE_OPENAI_API_KEY de container_apps.tf et
#    "azure-openai-api-key" de local.app_secret_names (main.tf).
# 3. Une fois /ask vérifié sans clé, couper les clés du compte, dans
#    azurerm_cognitive_account.ai ci-dessus :
#
#   local_auth_enabled = false
