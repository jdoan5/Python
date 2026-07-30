output "ecr_repository_url" {
  description = "Push the Stage 4 image here (deploy.sh does this)."
  value       = aws_ecr_repository.app.repository_url
}

output "app_url" {
  description = "Public URL of the deployed portal (HTTP — see README on TLS)."
  value       = "http://${aws_lb.app.dns_name}"
}

output "ecs_cluster" {
  value = aws_ecs_cluster.app.name
}

output "ecs_service" {
  value = aws_ecs_service.app.name
}

output "secret_id_anthropic" {
  description = "Use with: aws secretsmanager put-secret-value --secret-id ..."
  value       = aws_secretsmanager_secret.anthropic_api_key.name
}

output "secret_id_password" {
  value = aws_secretsmanager_secret.app_password.name
}
