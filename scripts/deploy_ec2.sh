#!/bin/bash

# Intelligent Analytics Platform - EC2 Deployment Script
# This script assumes you have Docker and Docker Compose installed on your EC2 instance.

echo "🚀 Starting EC2 Deployment Process..."

# 1. Update and install Docker if not present (Amazon Linux 2/2023)
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
fi

# 2. Build and run the container
echo "🏗️ Building and launching container..."
docker-compose up -d --build

echo "✅ Deployment successful!"
echo "🌐 Your app is running at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
echo "⚠️ Make sure to open port 8501 in your EC2 Security Group."
