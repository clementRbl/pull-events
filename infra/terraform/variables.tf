variable "subscription_id" {
  description = "Identifiant de l'abonnement Azure (dans terraform.tfvars, non versionné)."
  type        = string
}

variable "location" {
  description = "Région Azure de toutes les ressources."
  type        = string
  default     = "francecentral"
}

variable "name_suffix" {
  description = "Suffixe qui rend uniques les noms globaux (Key Vault, PostgreSQL, Foundry)."
  type        = string
  default     = "cr24"
}

variable "workstation_ip" {
  description = "Adresse IPv4 publique du poste de travail, seule IP autorisée sur PostgreSQL."
  type        = string
}

variable "api_image" {
  description = "Image de l'API (publique sur GHCR, sans secret)."
  type        = string
  default     = "ghcr.io/clementrbl/pull-events:0.2.1-azure"
}

variable "budget_amount" {
  description = "Budget mensuel (devise de facturation) au niveau de l'abonnement."
  type        = number
  default     = 10
}

variable "private_network" {
  description = "Phase réseau privé : VNet, Private Endpoint PostgreSQL, environnement Container Apps dans le VNet, règle 0.0.0.0 supprimée. Coût ≈ 7 €/mois (point de terminaison + zone DNS)."
  type        = bool
  default     = false
}

variable "enable_aks" {
  description = "Crée le cluster AKS (demande private_network = true). Coût ≈ 0,07 €/h : à détruire après la démonstration."
  type        = bool
  default     = false
}

variable "budget_email" {
  description = "Adresse qui reçoit les alertes de budget, en plus du rôle Owner."
  type        = string
}
