output "api_url" {
  description = "URL publique de l'API (Swagger sur /docs)."
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "postgres_fqdn" {
  description = "Nom DNS du serveur PostgreSQL."
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "azure_openai_endpoint" {
  description = "Point de terminaison Azure OpenAI."
  value       = "https://${azurerm_cognitive_account.ai.custom_subdomain_name}.openai.azure.com/"
}

output "key_vault_name" {
  description = "Coffre qui contient tous les secrets."
  value       = azurerm_key_vault.main.name
}

output "aks_name" {
  description = "Nom du cluster AKS (vide si enable_aks = false)."
  value       = var.enable_aks ? azurerm_kubernetes_cluster.main[0].name : null
}

output "aks_kv_identity_client_id" {
  description = "Client ID de l'identité de l'add-on CSI (valeur secrets.keyVault.clientId du chart)."
  value       = var.enable_aks ? azurerm_kubernetes_cluster.main[0].key_vault_secrets_provider[0].secret_identity[0].client_id : null
}

output "aks_ingress_fqdn" {
  description = "Nom DNS de l'IP publique de l'ingress AKS."
  value       = var.enable_aks ? azurerm_public_ip.ingress[0].fqdn : null
}

output "tenant_id" {
  description = "Tenant Entra ID (valeur secrets.keyVault.tenantId du chart)."
  value       = data.azurerm_client_config.current.tenant_id
}
