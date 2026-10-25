# Budget mensuel au niveau de l'abonnement : e-mail aux propriétaires et à une
# adresse explicite (audit 6.1). Un budget alerte, il ne bloque rien.

resource "azurerm_consumption_budget_subscription" "main" {
  name            = "budget-pull-events"
  subscription_id = "/subscriptions/${var.subscription_id}"
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
    start_date = "2026-10-01T00:00:00Z"
    end_date   = "2027-10-31T00:00:00Z"
  }

  notification {
    operator       = "GreaterThan"
    threshold      = 50
    threshold_type = "Actual"
    contact_roles  = ["Owner"]
    contact_emails = [var.budget_email]
  }
  notification {
    operator       = "GreaterThan"
    threshold      = 80
    threshold_type = "Actual"
    contact_roles  = ["Owner"]
    contact_emails = [var.budget_email]
  }
  notification {
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Actual"
    contact_roles  = ["Owner"]
    contact_emails = [var.budget_email]
  }
  notification {
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Forecasted"
    contact_roles  = ["Owner"]
    contact_emails = [var.budget_email]
  }
}
