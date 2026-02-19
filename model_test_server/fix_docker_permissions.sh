#!/bin/bash
# Quick fix for Docker permission issues

echo "="*70
echo "Docker Permission Fix"
echo "="*70
echo ""

# Check if user is in docker group
if groups | grep -q docker; then
    echo "✓ User is in docker group"
else
    echo "✗ User is NOT in docker group"
    echo ""
    echo "Adding user to docker group..."
    sudo usermod -aG docker $USER
    echo "✓ User added to docker group"
    echo ""
    echo "⚠ You need to start a new shell session for this to take effect!"
fi

echo ""
echo "Current groups:"
groups

echo ""
echo "To fix permission issues, choose one:"
echo ""
echo "Option 1: Start new shell with docker group (recommended)"
echo "  newgrp docker"
echo "  # Then test: docker run hello-world"
echo ""
echo "Option 2: Log out and log back in"
echo ""
echo "Option 3: Use sudo (temporary workaround)"
echo "  sudo docker run hello-world"
echo ""
echo "Option 4: Fix socket permissions (not recommended, security risk)"
echo "  sudo chmod 666 /var/run/docker.sock"
echo "  # This allows all users to access Docker - use with caution!"
echo ""

# Check if we can access docker socket
if [ -w /var/run/docker.sock ] 2>/dev/null; then
    echo "✓ Docker socket is writable"
elif [ -r /var/run/docker.sock ] 2>/dev/null; then
    echo "⚠ Docker socket is readable but not writable"
else
    echo "✗ Cannot access Docker socket"
fi

echo ""
echo "="*70
