variable "region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "ap-south-1" # Mumbai
}

variable "instance_type" {
  description = "EC2 instance type (t3.micro is free-plan eligible)"
  type        = string
  default     = "t3.micro"
}

variable "repo_url" {
  description = "Git repository the instance clones and runs"
  type        = string
  default     = "https://github.com/Mickeyzer/Model-Guard.git"
}
