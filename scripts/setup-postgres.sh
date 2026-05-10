#!/usr/bin/env bash
# Knowledge Palace - PostgreSQL Setup Script
# Run this with sudo to install and configure PostgreSQL + pgvector

set -euo pipefail

echo "🏛️  Knowledge Palace — PostgreSQL Setup"
echo "========================================"
echo ""

# Check if running with sudo/root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script requires sudo. Run: sudo bash scripts/setup-postgres.sh"
    exit 1
fi

# Install PostgreSQL + pgvector
echo "📦 Installing PostgreSQL 16 + pgvector..."
apt update -qq
apt install -y postgresql-16 postgresql-16-pgvector postgresql-client-16

# Start PostgreSQL
echo "🔧 Starting PostgreSQL..."
pg_ctlcluster 16 main start 2>/dev/null || true

# Create database and user
echo "👤 Creating database user 'kp'..."
su - postgres -c "psql -c \"CREATE USER kp WITH PASSWORD 'kp';\"" 2>/dev/null || echo "User already exists"

echo "🗄️  Creating database 'knowledge_palace'..."
su - postgres -c "psql -c \"CREATE DATABASE knowledge_palace OWNER kp;\"" 2>/dev/null || echo "Database already exists"

echo "🔌 Enabling pgvector extension..."
su - postgres -c "psql -d knowledge_palace -c \"CREATE EXTENSION IF NOT EXISTS vector;\""

echo ""
echo "✅ PostgreSQL is ready!"
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: knowledge_palace"
echo "  User: kp"
echo "  Password: kp"
echo ""
echo "Next steps:"
echo "  cd ~/projects/knowledge-palace"
echo "  kp init-config        # Create config.toml"
echo "  kp ingest --source all  # Ingest your documents"
echo "  kp serve              # Start API server"
