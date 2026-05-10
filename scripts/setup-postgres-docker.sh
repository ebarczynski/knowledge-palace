#!/usr/bin/env bash
# Knowledge Palace - Docker-based PostgreSQL Setup (no root needed)
# Requires Docker Desktop running with WSL integration enabled

set -euo pipefail

echo "🏛️  Knowledge Palace — Docker PostgreSQL Setup"
echo "==============================================="
echo ""

# Check docker
if ! command -v docker &>/dev/null; then
    echo "❌ Docker not found. Start Docker Desktop and enable WSL integration."
    echo "   Docker Desktop > Settings > Resources > WSL Integration"
    exit 1
fi

echo "🐳 Starting PostgreSQL + pgvector container..."
docker run -d \
    --name kp-postgres \
    -e POSTGRES_DB=knowledge_palace \
    -e POSTGRES_USER=kp \
    -e POSTGRES_PASSWORD=kp \
    -v kp-pgdata:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg16

echo "⏳ Waiting for PostgreSQL to start..."
sleep 5

# Verify
docker exec kp-postgres pg_isready -U kp
echo ""
echo "✅ PostgreSQL is ready via Docker!"
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: knowledge_palace"
echo "  User: kp"
echo "  Password: kp"
echo ""
echo "To stop:  docker stop kp-postgres"
echo "To start: docker start kp-postgres"
echo ""
echo "Next steps:"
echo "  cd ~/projects/knowledge-palace"
echo "  kp ingest --source all  # Ingest your documents"
echo "  kp serve              # Start API server"
