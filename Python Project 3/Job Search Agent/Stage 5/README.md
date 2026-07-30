# Stage 5 — Terraform on AWS (ECS Fargate + ALB)

Provisions the Stage 4 container on AWS with infrastructure as code:

```
ECR                       private image registry (deploy.sh pushes here)
ECS Fargate               runs the container (1 task, 1 vCPU / 2 GB)
Application Load Balancer WebSocket-capable front door, health checks
Secrets Manager           ANTHROPIC_API_KEY + APP_PASSWORD, injected at task start
IAM                       task execution role scoped to exactly two secret ARNs
CloudWatch Logs           container logs, 7-day retention
```

The Stage 4 image runs **unchanged** — laptop → Fly → AWS, same container.

## Why ECS + ALB and not App Runner

The first draft of this stage used AWS App Runner. The adversarial review
caught two disqualifiers before anything was deployed:

1. **App Runner doesn't support WebSockets** — and Streamlit's UI runs
   entirely over one. The service would deploy green (the HTTP health check
   passes) and then hang forever at "Please wait…" in the browser.
2. **App Runner closed to new customers on April 30, 2026** — this account
   couldn't have created the service at all.

ECS Fargate behind an ALB is the classic answer: full WebSocket support
(`idle_timeout = 3600` so a multi-minute pipeline run doesn't drop the
socket), and the more valuable resume pattern anyway.

## Costs — read this first

This stack does **not** scale to zero:

| Piece | Idle cost (us-east-1, approx) |
|---|---|
| ALB | ~$16/mo + LCU |
| Fargate 1 vCPU / 2 GB | ~$36/mo |
| **Total left running** | **~$50/mo** |

Your $100 signup credits cover ~2 months of always-on — but the working
habit is: **demo, then destroy.**

```bash
terraform destroy                      # full teardown

# Cheaper pause (keeps ALB ~$16/mo, drops Fargate cost to zero):
aws ecs update-service --cluster job-search-agent --service job-search-agent --desired-count 0
# resume:
aws ecs update-service --cluster job-search-agent --service job-search-agent --desired-count 1
```

The zero-spend budget you set will email you when credits are exhausted.

## Security model — honest version

The ALB URL is **public internet-facing**. The only gate is the Stage 4
shared password — no rate limiting, no lockout. And without a custom domain
there's no TLS (ACM certificates require a domain), so the login password
travels over plain HTTP. Acceptable for a short-lived demo **if the password
is unique and low-value** (never reuse a real one). The production
extension: Route 53 domain + ACM cert + HTTPS listener + HTTP→HTTPS
redirect.

## Known limitation — no persistent storage

No volume is attached: `/data` (tracker DB + approved drafts) resets on
every deploy or task replacement. Deliberate scope cut and a good interview
talking point (compute/state separation). Stateful options: Fly.io with a
volume (Stage 4), or EFS/RDS here (future work).

## Deploy sequence — first time AND after every destroy

Order matters: the service can't start from an empty registry or empty
secrets. **After a `terraform destroy` you must repeat ALL of this** —
destroy deletes the pushed images (`force_delete`) and the secret values
(0-day recovery window). A bare `terraform apply` from empty state will
create a service that can never start.

```bash
cd "Stage 5"

# 0. Sanity: you should be jdoan-admin in us-east-1
aws sts get-caller-identity

# 1. Init providers
terraform init

# 2. Create just the registry + secret containers
terraform apply -target=aws_ecr_repository.app \
                -target=aws_secretsmanager_secret.anthropic_api_key \
                -target=aws_secretsmanager_secret.app_password

# 3. Build (linux/amd64) and push the image — Docker Desktop must be running
./deploy.sh

# 4. Set the secret VALUES (never in .tf files — see secrets.tf for why)
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw secret_id_anthropic)" --secret-string 'sk-ant-...'
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw secret_id_password)" --secret-string 'choose-a-password'

# 5. Create everything else (~3-5 minutes to a healthy target)
terraform apply

# 6. Open it
terraform output app_url
```

Sign in with the password from step 4. Note the container ships **without
your experience inventory** (personal data, dockerignored) — the tracker
dashboard works immediately; the tailor page needs an inventory, which on
AWS means either baking a sanitized sample into the image or treating
Fly/local as the pipeline's home. See "Known limitation".

## Updating the app

```bash
./deploy.sh    # build + push + roll the ECS service onto the new image
```

**Rotating a secret**: ECS resolves secrets at task START. After
`put-secret-value`, force a redeploy or the running task keeps the old
value:

```bash
aws ecs update-service --cluster "$(terraform output -raw ecs_cluster)" \
  --service "$(terraform output -raw ecs_service)" --force-new-deployment
```

## Files

| File | What |
|---|---|
| `versions.tf` | Terraform ≥1.9, AWS provider ~>6.0, default tags |
| `variables.tf` | region / name / tag / cpu / memory |
| `ecr.tf` | repository + keep-5-images lifecycle policy |
| `secrets.tf` | secret *containers* only — values via CLI, never in state |
| `network.tf` | default VPC lookups + two security groups (ALB-only ingress to the task) |
| `alb.tf` | ALB (3600s idle timeout for WebSockets), target group + `/_stcore/health` check |
| `iam.tf` | task execution role: ECR pull, logs, `GetSecretValue` on exactly 2 ARNs |
| `ecs.tf` | cluster, task definition (secrets injection, awslogs), service |
| `outputs.tf` | app URL, ECR URL, cluster/service names, secret ids |
| `deploy.sh` | amd64 build (`--provenance=false`), ECR push, service roll |
