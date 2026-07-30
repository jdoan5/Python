# ECS on Fargate — runs the Stage 4 container. Secrets are resolved by the
# execution role at task START (rotating a secret needs a redeploy — see
# README). Logs land in CloudWatch with short retention to cap cost.
#
# KNOWN LIMITATION (same as App Runner would have had): no persistent volume
# here — /data (tracker DB + drafts) resets on every deploy/restart. Fly
# (Stage 4) remains the stateful deployment; EFS or RDS is the future-work
# answer on AWS.

resource "aws_ecs_cluster" "app" {
  name = var.app_name
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "portal"
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true

      portMappings = [
        { containerPort = 8501, protocol = "tcp" }
      ]

      secrets = [
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic_api_key.arn },
        { name = "APP_PASSWORD", valueFrom = aws_secretsmanager_secret.app_password.arn },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "portal"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "app" {
  name            = var.app_name
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  # Streamlit needs ~20-30s to boot before health checks should count.
  health_check_grace_period_seconds = 60

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true # default VPC has no NAT; task needs egress for ECR pull + Anthropic API
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "portal"
    container_port   = 8501
  }

  depends_on = [
    aws_lb_listener.http,
    aws_iam_role_policy.read_app_secrets,
    aws_iam_role_policy_attachment.task_execution_managed,
  ]
}
