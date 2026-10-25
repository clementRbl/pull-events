# Journaux, environnement Container Apps et l'application API.

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-pull-events"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
  # Audit 7.2 : sans plafond, inonder l'API de 401 gonfle la facture d'ingestion
  # (5 Go gratuits par mois, puis ≈ 2,5 €/Go). Au-delà de 0,5 Go par jour,
  # Azure arrête d'ingérer jusqu'au lendemain.
  daily_quota_gb = 0.5
  # Valeur par défaut d'Azure, écrite explicitement pour un plan stable.
  local_authentication_enabled = true
}

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-pull-events"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  logs_destination           = "log-analytics"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags

  # Un environnement ne peut pas rejoindre un VNet après coup : activer
  # var.private_network le recrée (et l'application avec lui, nouvelle URL).
  infrastructure_subnet_id = var.private_network ? azurerm_subnet.aca[0].id : null

  # Profil Consommation : facturé à l'usage, avec la franchise mensuelle gratuite.
  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
  }
}

resource "azurerm_container_app" "api" {
  name                         = "ca-pull-events-api"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  workload_profile_name        = "Consumption"
  revision_mode                = "Single"
  max_inactive_revisions       = 100
  tags                         = local.tags

  identity {
    type = "SystemAssigned"
  }

  # Le provider exige identity = "System" mais Azure renvoie "system" : sans
  # cette ligne, chaque plan voudrait réécrire les cinq secrets à l'identique.
  # Pour ajouter ou retirer une référence Key Vault : modifier le code ET
  # exécuter `az containerapp secret set` (voir docs/azure/terraform.md).
  lifecycle {
    ignore_changes = [secret]
  }

  # Chaque secret est une référence vers le Key Vault, lue par l'identité
  # managée : Container Apps ne stocke que l'URI, pas la valeur.
  secret {
    name                = "api-key"
    identity            = "System"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/api-key"
  }
  secret {
    name                = "azure-openai-api-key"
    identity            = "System"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/azure-openai-api-key"
  }
  secret {
    name                = "database-url"
    identity            = "System"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/database-url"
  }
  secret {
    name                = "langfuse-public-key"
    identity            = "System"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/langfuse-public-key"
  }
  secret {
    name                = "langfuse-secret-key"
    identity            = "System"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/langfuse-secret-key"
  }

  ingress {
    external_enabled           = true
    target_port                = 8000
    transport                  = "auto"
    allow_insecure_connections = false

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    # Mise à zéro sans trafic : la démo ne coûte rien quand personne ne l'utilise.
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "ca-pull-events-api"
      image  = var.api_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "ENABLE_REBUILD"
        value = "false"
      }
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }
      env {
        name        = "API_KEY"
        secret_name = "api-key"
      }
      env {
        name        = "LANGFUSE_PUBLIC_KEY"
        secret_name = "langfuse-public-key"
      }
      env {
        name        = "LANGFUSE_SECRET_KEY"
        secret_name = "langfuse-secret-key"
      }
      env {
        name  = "LANGFUSE_BASE_URL"
        value = "https://cloud.langfuse.com"
      }
      env {
        name  = "LLM_PROVIDER"
        value = "azure_openai"
      }
      env {
        name  = "VECTOR_TABLE"
        value = "event_chunks"
      }
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = "https://${azurerm_cognitive_account.ai.custom_subdomain_name}.openai.azure.com/"
      }
      env {
        name        = "AZURE_OPENAI_API_KEY"
        secret_name = "azure-openai-api-key"
      }
      env {
        name  = "LLM_MODEL"
        value = azurerm_cognitive_deployment.gpt.name
      }
      env {
        name  = "EMBEDDING_MODEL"
        value = azurerm_cognitive_deployment.embeddings.name
      }
      env {
        name  = "EMBEDDING_SIZE"
        value = "1536"
      }
      # Prix en dollars par million de tokens (Standard régional France Central).
      env {
        name  = "LLM_INPUT_PRICE_PER_MTOK"
        value = "0.484"
      }
      env {
        name  = "LLM_OUTPUT_PRICE_PER_MTOK"
        value = "1.936"
      }
      env {
        name  = "EMBEDDING_PRICE_PER_MTOK"
        value = "0.022"
      }
    }
  }
}
