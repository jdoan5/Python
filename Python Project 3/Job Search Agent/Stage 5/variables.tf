variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Base name for all resources."
  type        = string
  default     = "job-search-agent"
}

variable "image_tag" {
  description = "Tag of the image in ECR that App Runner should run."
  type        = string
  default     = "latest"
}

variable "cpu" {
  description = "App Runner vCPU units (256|512|1024|2048|4096)."
  type        = string
  default     = "1024"
}

variable "memory" {
  description = "App Runner memory in MB (512|1024|2048|3072|4096|...)."
  type        = string
  default     = "2048"
}
