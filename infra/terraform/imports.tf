# Import déclaratif des ressources créées à la main avec la CLI (24/10/2026).
# Au premier `terraform apply`, Terraform adopte chaque ressource dans l'état au
# lieu de la recréer. Ces blocs restent dans le code : ils documentent d'où
# vient l'infrastructure, et sont ignorés une fois l'import fait.

locals {
  rg_id = "/subscriptions/${var.subscription_id}/resourceGroups/rg-pull-events"
  kv_id = "${local.rg_id}/providers/Microsoft.KeyVault/vaults/kv-pull-events-${var.name_suffix}"
  ai_id = "${local.rg_id}/providers/Microsoft.CognitiveServices/accounts/ais-pull-events-${var.name_suffix}"
  pg_id = "${local.rg_id}/providers/Microsoft.DBforPostgreSQL/flexibleServers/psql-pull-events-${var.name_suffix}"
}

import {
  to = azurerm_resource_group.main
  id = local.rg_id
}

import {
  to = azurerm_key_vault.main
  id = local.kv_id
}

import {
  to = azurerm_role_assignment.kv_secrets_officer_me
  id = "${local.kv_id}/providers/Microsoft.Authorization/roleAssignments/f01b5a51-bbc1-4543-b8f6-23256866bde9"
}

import {
  to = azurerm_cognitive_account.ai
  id = local.ai_id
}

import {
  to = azurerm_cognitive_deployment.gpt
  id = "${local.ai_id}/deployments/gpt-4.1-mini"
}

import {
  to = azurerm_cognitive_deployment.embeddings
  id = "${local.ai_id}/deployments/text-embedding-3-small"
}

import {
  to = azurerm_postgresql_flexible_server.main
  id = local.pg_id
}

import {
  to = azurerm_postgresql_flexible_server_database.pullevents
  id = "${local.pg_id}/databases/pullevents"
}

import {
  to = azurerm_postgresql_flexible_server_configuration.extensions
  id = "${local.pg_id}/configurations/azure.extensions"
}

import {
  to = azurerm_postgresql_flexible_server_firewall_rule.workstation
  id = "${local.pg_id}/firewallRules/FirewallIPAddress_2026-10-24_23-18-40"
}

import {
  to = azurerm_postgresql_flexible_server_firewall_rule.allow_azure_services[0]
  id = "${local.pg_id}/firewallRules/allow-azure-services"
}

import {
  to = azurerm_log_analytics_workspace.main
  id = "${local.rg_id}/providers/Microsoft.OperationalInsights/workspaces/log-pull-events"
}

import {
  to = azurerm_container_app_environment.main
  id = "${local.rg_id}/providers/Microsoft.App/managedEnvironments/cae-pull-events"
}

import {
  to = azurerm_container_app.api
  id = "${local.rg_id}/providers/Microsoft.App/containerApps/ca-pull-events-api"
}

import {
  to = azurerm_consumption_budget_subscription.main
  id = "/subscriptions/${var.subscription_id}/providers/Microsoft.Consumption/budgets/budget-pull-events"
}
