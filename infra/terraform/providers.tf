provider "azurerm" {
  features {
    key_vault {
      # Les secrets ne sont pas gérés ici ; ne jamais purger le coffre par accident.
      purge_soft_delete_on_destroy = false
    }
  }

  subscription_id = var.subscription_id

  # Les fournisseurs de ressources (Microsoft.App, Microsoft.Network, ...)
  # sont enregistrés une fois avec `az provider register` : voir la doc.
  resource_provider_registrations = "none"
}
