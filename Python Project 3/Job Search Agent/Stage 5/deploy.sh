#!/usr/bin/env bash
# Build the Stage 4 image for linux/amd64 and push it to ECR, then roll the
# ECS service onto it (if the service exists yet).
#
# Usage (from the Stage 5 folder, after the targeted terraform apply):
#   ./deploy.sh
#
# Requires: docker (daemon running), aws cli configured, terraform state
# with the ECR repo created.

set -euo pipefail

cd "$(dirname "$0")"

REPO_URL="$(terraform output -raw ecr_repository_url 2>/dev/null || true)"
if [ -z "$REPO_URL" ]; then
  echo "error: could not read ecr_repository_url from terraform output." >&2
  echo "Run the targeted apply from the README first." >&2
  exit 1
fi

REGISTRY="${REPO_URL%%/*}"                     # <account>.dkr.ecr.<region>.amazonaws.com
REGION="$(echo "$REGISTRY" | cut -d. -f4)"
TAG="${1:-latest}"

echo "==> Logging in to $REGISTRY"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "==> Building for linux/amd64 (Fargate runs amd64; this Mac is arm64)"
# --provenance=false: BuildKit attestations create an OCI image index that
# some AWS services fail to pull; a plain single-manifest image is safest.
docker build \
  --platform linux/amd64 \
  --provenance=false \
  -f "../Stage 4/Dockerfile" \
  -t "$REPO_URL:$TAG" \
  ..

echo "==> Pushing $REPO_URL:$TAG"
docker push "$REPO_URL:$TAG"

# ECS does NOT auto-redeploy when a tag is re-pushed — roll the service if
# it exists (first-time setup: it doesn't yet; the full apply creates it).
CLUSTER="$(terraform output -raw ecs_cluster 2>/dev/null || true)"
SERVICE="$(terraform output -raw ecs_service 2>/dev/null || true)"
if [ -n "$CLUSTER" ] && [ -n "$SERVICE" ]; then
  echo "==> Rolling ECS service $SERVICE onto the new image"
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --force-new-deployment --region "$REGION" --no-cli-pager > /dev/null
  echo "==> Deployment started; watch: aws ecs describe-services --cluster $CLUSTER --services $SERVICE"
else
  echo "==> ECS service not created yet — continue with the README (secret values, then terraform apply)."
fi
