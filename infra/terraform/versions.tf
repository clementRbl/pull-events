# Versions figées : Terraform 1.17 et le provider azurerm 5.x.
# Le backend (où vit l'état) est déclaré dans backend.tf.
terraform {
  required_version = ">= 1.17.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.11"
    }
  }

}
