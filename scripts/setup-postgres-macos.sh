#!/usr/bin/env bash
# Knowledge Palace - local PostgreSQL setup for macOS (Homebrew + Apple Silicon).
#
# Idempotent: safe to run repeatedly. Creates the kp user, knowledge_palace
# database, and pgvector extension only if they don't already exist.
#
# Usage:
#   bash scripts/setup-postgres-macos.sh

set -euo pipefail

echo "🏛️  Knowledge Palace — local PostgreSQL setup (macOS)"
echo "====================================================="
echo ""

# 1. Homebrew
if ! command -v brew &>/dev/null; then
    echo "❌ Homebrew not found. Install it first: https://brew.sh"
    exit 1
fi

# 2. PostgreSQL 17 + pgvector
if ! brew list --formula 2>/dev/null | grep -q "^postgresql@17$"; then
    echo "📦 Installing PostgreSQL 17 + pgvector (this can take a few minutes)..."
    brew install postgresql@17 pgvector
else
    echo "✓ postgresql@17 already installed"
fi
brew list --formula 2>/dev/null | grep -q "^pgvector$" || brew install pgvector

# Homebrew's postgresql@17 is keg-only: its bin is not on PATH by default.
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# 3. Start the service (and keep it running across reboots)
if ! pg_isready -q 2>/dev/null; then
    echo "🔧 Starting postgresql@17 service..."
    brew services start postgresql@17
    sleep 3
fi
pg_isready && echo "✓ PostgreSQL is accepting connections"

# 4. Create the kp role (idempotent)
if psql postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='kp'" | grep -q 1; then
    echo "✓ role 'kp' already exists"
else
    psql postgres -c "CREATE USER kp WITH PASSWORD 'kp';"
    echo "✓ created role 'kp' (password 'kp' — local dev only)"
fi

# 5. Create the database (idempotent)
if psql postgres -tAc "SELECT 1 FROM pg_database WHERE datname='knowledge_palace'" | grep -q 1; then
    echo "✓ database 'knowledge_palace' already exists"
else
    createdb -O kp knowledge_palace
    echo "✓ created database 'knowledge_palace'"
fi

# 6. Enable pgvector (idempotent)
psql -d knowledge_palace -tAc "SELECT 1 FROM pg_extension WHERE extname='vector'" | grep -q 1 \
    || psql -d knowledge_palace -c "CREATE EXTENSION IF NOT EXISTS vector;"

# A registered extension is not proof that the library loads. A pgvector
# upgrade that no longer ships a build for this PostgreSQL major leaves the
# catalog row in place while every query against the chunks table fails with
# 'could not access file "$libdir/vector"'. Exercise the type instead.
if ! psql -d knowledge_palace -tAc "SELECT '[1,2,3]'::vector <=> '[3,2,1]'::vector" > /dev/null; then
    echo "❌ pgvector is registered but its library cannot be loaded by this PostgreSQL."
    echo "   Rebuild pgvector for the running major version (no sudo needed under"
    echo "   Homebrew). PG_SYSROOT is required because Homebrew's pg_config points"
    echo "   at a CommandLineTools SDK that may not exist on this machine:"
    echo "     git clone --branch v0.8.2 https://github.com/pgvector/pgvector && cd pgvector"
    echo '     make PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config PG_SYSROOT="$(xcrun --show-sdk-path)"'
    echo '     make install PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config PG_SYSROOT="$(xcrun --show-sdk-path)"'
    echo "   (Use the version the database has registered: SELECT extversion FROM pg_extension WHERE extname='vector'.)"
    echo "   Then run this script again."
    exit 1
fi
echo "✓ pgvector extension ready"

echo ""
echo "✅ Local database ready."
echo ""
echo "Next steps (from the project root):"
echo "  uv sync                                # python environment"
echo "  uv run kp ingest --source all          # ingest Calibre + files"
echo "  uv run kp serve                        # API on http://127.0.0.1:8080"
echo "  uv run kp search 'your query'          # CLI search"
