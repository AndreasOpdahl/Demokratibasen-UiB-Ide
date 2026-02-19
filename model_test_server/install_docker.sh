#!/bin/bash
# Install Docker on Ubuntu
# Based on official Docker installation instructions

set -e

echo "Installing Docker..."

sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Test installation
sudo docker run hello-world

# Add current user to docker group
sudo usermod -aG docker $USER

echo ""
echo "✓ Docker installed successfully!"
echo ""
echo "⚠ IMPORTANT: Group changes require a new shell session."
echo ""
echo "To apply group changes immediately, choose one:"
echo "  1. Run: newgrp docker"
echo "     (This starts a new shell with the docker group)"
echo ""
echo "  2. Log out and log back in"
echo ""
echo "  3. Or use sudo for now:"
echo "     sudo docker run hello-world"
echo ""
echo "After applying group changes, verify with:"
echo "  docker run hello-world"
