# Secret CONTAINERS only — the values are set out-of-band with the CLI:
#
#   aws secretsmanager put-secret-value \
#     --secret-id job-search-agent/anthropic-api-key --secret-string 'sk-ant-...'
#   aws secretsmanager put-secret-value \
#     --secret-id job-search-agent/app-password --secret-string 'your-password'
#
# Never put secret values in .tf or .tfvars files: Terraform state stores
# every attribute in plaintext, and state files end up in backups/laptops.
# Keeping values out of Terraform entirely means the state only ever holds
# the secret's ARN, not its content.

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name        = "${var.app_name}/anthropic-api-key"
  description = "Anthropic API key for the Job Search Agent pipeline."

  # Demo-friendly: destroy removes the secret immediately instead of the
  # 7-30 day recovery window (which would block re-creating it on re-apply).
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "app_password" {
  name        = "${var.app_name}/app-password"
  description = "Shared password for the portal's login gate."

  recovery_window_in_days = 0
}
