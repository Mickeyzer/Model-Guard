#!/bin/bash
# EC2 user data: runs once as root on first boot. Log: /var/log/cloud-init-output.log
set -euxo pipefail

# t3.micro has 1 GB RAM, too little to build the images; add 2 GB swap
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Docker Engine + Compose plugin
curl -fsSL https://get.docker.com | sh
usermod -aG docker ubuntu

git clone ${repo_url} /opt/modelguard
cd /opt/modelguard

# Bring the stack back up after a reboot
cat > docker-compose.override.yml <<'EOF'
services:
  protected_model:
    restart: unless-stopped
  middleware:
    restart: unless-stopped
  dashboard:
    restart: unless-stopped
EOF

docker compose up -d --build
touch /opt/modelguard/.bootstrap-done
