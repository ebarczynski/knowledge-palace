#!/usr/bin/env bash
# Knowledge Palace - Docker-based PostgreSQL Setup (no root needed)
# Requires a running Docker daemon (Docker Desktop on macOS/Windows, dockerd on Linux)

set -euo pipefail

echo "🏛️  Knowledge Palace — Docker PostgreSQL Setup"
echo "==============================================="
echo ""

# Check docker
if ! command -v docker &>/dev/null; then
    echo "❌ Docker not found. Install it and start the daemon first."
    exit 1
fi
if ! docker info > /dev/null 2>&1; then
    echo "❌ The Docker daemon is not running. Start Docker Desktop (or dockerd) first."
    exit 1
fi

echo "🐳 Starting PostgreSQL + pgvector container..."
if docker container inspect kp-postgres > /dev/null 2>&1; then
    # Re-running the script must not fail on "container name already in use".
    docker start kp-postgres
else
    docker run -d \
        --name kp-postgres \
        -e POSTGRES_DB=knowledge_palace \
        -e POSTGRES_USER=kp \
        -e POSTGRES_PASSWORD=kp \
        -v kp-pgdata:/var/lib/postgresql/data \
        -p 5432:5432 \
        pgvector/pgvector:pg16
fi

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
