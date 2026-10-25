# Où vit l'état Terraform.
#
# Cible : un compte de stockage Azure créé une fois par bootstrap-backend.sh.
# Le verrou est un bail (lease) sur le blob, pris par Terraform pendant chaque
# plan/apply ; l'authentification passe par Entra ID (az login), sans clé.
# En attendant la décision sur les coûts, l'état reste local (fichier ignoré
# par git). Pour migrer : remplacer le bloc local par le bloc azurerm, puis
# `terraform init -migrate-state`.
terraform {
  backend "local" {}

  # backend "azurerm" {
  #   resource_group_name  = "rg-pull-events-tfstate"
  #   storage_account_name = "stpulleventstfcr24"
  #   container_name       = "tfstate"
  #   key                  = "pull-events.tfstate"
  #   use_azuread_auth     = true
  # }
}
