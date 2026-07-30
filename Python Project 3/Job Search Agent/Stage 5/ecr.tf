# Private image registry. The Stage 4 image is pushed here by deploy.sh;
# App Runner pulls from it (auto-deploy on every push of var.image_tag).

resource "aws_ecr_repository" "app" {
  name = var.app_name

  image_scanning_configuration {
    scan_on_push = true
  }

  # A learning/demo repo: allow `terraform destroy` even with images present.
  force_delete = true
}

# Keep the repo from accumulating storage cost: retain only the 5 newest images.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only the 5 most recent images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      }
    ]
  })
}
