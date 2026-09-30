# ModelGuard on a single AWS EC2 instance (free-plan eligible t3.micro).
#
#   terraform init
#   terraform apply      # creates everything, prints the URLs
#   terraform destroy    # removes everything
#
# Only the middleware (8000) and dashboard (8501) are reachable from the
# internet. The protected model (8001) stays private, so clients cannot
# bypass the middleware. SSH (22) is open to the deployer's IP only.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws   = { source = "hashicorp/aws", version = ">= 5.0" }
    tls   = { source = "hashicorp/tls", version = ">= 4.0" }
    local = { source = "hashicorp/local", version = ">= 2.4" }
    http  = { source = "hashicorp/http", version = ">= 3.4" }
  }
}

provider "aws" {
  region = var.region
}

# Latest Ubuntu 24.04 LTS (x86_64) published by Canonical
data "aws_ssm_parameter" "ubuntu" {
  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

locals {
  my_cidr = "${trimspace(data.http.my_ip.response_body)}/32"
}

resource "tls_private_key" "ssh" {
  algorithm = "ED25519"
}

resource "aws_key_pair" "modelguard" {
  key_name   = "modelguard-key"
  public_key = tls_private_key.ssh.public_key_openssh
}

resource "local_sensitive_file" "ssh_key" {
  content         = tls_private_key.ssh.private_key_openssh
  filename        = "${path.module}/modelguard-key.pem"
  file_permission = "0600"
}

resource "aws_security_group" "modelguard" {
  name        = "modelguard-sg"
  description = "ModelGuard: public API + dashboard, SSH from deployer only"

  ingress {
    description = "Middleware API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Streamlit dashboard"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH from deployer IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.my_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "modelguard" {
  ami                    = data.aws_ssm_parameter.ubuntu.value
  instance_type          = var.instance_type
  key_name               = aws_key_pair.modelguard.key_name
  vpc_security_group_ids = [aws_security_group.modelguard.id]
  user_data              = templatefile("${path.module}/bootstrap.sh", { repo_url = var.repo_url })

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  # IMDSv2 only
  metadata_options {
    http_tokens = "required"
  }

  tags = { Name = "modelguard" }
}
