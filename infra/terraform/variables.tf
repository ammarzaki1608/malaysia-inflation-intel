variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-southeast1" # Singapore — closest to Penang, lowest latency
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "alert_email" {
  description = "Email address notified when the monthly pipeline job fails. Leave empty to skip creating alerts."
  type        = string
  default     = ""
}
