#!/usr/bin/env bash
# Knowledge Palace — migrate to another Mac over the local network.
#
# Pushes project code, the Calibre library, and the PostgreSQL database
# (with embeddings) to a target Mac and verifies search works there.
#
# Usage:
#   ./scripts/migrate-to-host.sh <user>@<host> [stage]
#
# Stages (run in order; each is idempotent/resumable):
#   all       everything below (default)
#   code      rsync project (excludes .venv/.git/caches)
#   db        pg_dump -> scp -> parallel pg_restore on target
#   verify    chunk count + kp search smoke test on target
#   books     rsync the Calibre library (9+ GB, run last / overnight)
#
# Prerequisites on the TARGET Mac (one-time):
#   brew install postgresql@17 pgvector uv
#   brew services start postgresql@17
#   psql postgres -c "CREATE USER kp WITH PASSWORD 'kp';"
#   createdb -O kp knowledge_palace
#   psql -d knowledge_palace -c "CREATE EXTENSION IF NOT EXISTS vector;"
#
# NOTE: pg_restore -j (parallel) cannot read from a pipe, so the DB is dumped
# to a local file, scp'd, then restored from the file on the target.

set -euo pipefail

DEST="${1:?usage: $0 <user>@<host> [stage]}"
STAGE="${2:-all}"

SRC_HOME="$HOME"
SRC_PROJECT="$SRC_HOME/Documents/knowledge-palace"
SRC_BOOKS="$SRC_HOME/Documents/Books"
DUMP_FILE="/tmp/knowledge_palace.dump"

# Non-interactive ssh shells don't source .zshrc, so Homebrew's bin is missing
# from PATH there. Prefix every remote command with this.
RPATH='PATH=/opt/homebrew/bin:/usr/local/bin:$PATH'

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; exit 1; }

stage_code() {
    log "Stage: project code -> $DEST"
    rsync -az --info=progress2 \
        --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
        --exclude='.zcode' \
        "$SRC_PROJECT/" "$DEST:/Users/${DEST%%@*}/Documents/knowledge-palace/"
    ok "project synced"
}

stage_db() {
    log "Stage: database -> $DEST"
    log "  dumping (compressed, ~700MB)..."
    PGPASSWORD=kp pg_dump -h localhost -U kp -Fc -d knowledge_palace -f "$DUMP_FILE"
    ok "dump written: $(du -h "$DUMP_FILE" | cut -f1)"

    log "  copying dump..."
    scp -q "$DUMP_FILE" "$DEST:/tmp/knowledge_palace.dump"
    ok "copied"

    log "  restoring on target (parallel)..."
    ssh "$DEST" "$RPATH PGPASSWORD=kp pg_restore -h localhost -U kp -d knowledge_palace -j 4 --no-owner --no-privileges /tmp/knowledge_palace.dump && rm /tmp/knowledge_palace.dump"
    ok "restored"
    rm -f "$DUMP_FILE"
}

stage_verify() {
    log "Stage: verify on $DEST"

    chunks=$(ssh "$DEST" "$RPATH PGPASSWORD=kp psql -h localhost -U kp -d knowledge_palace -t -A -c 'SELECT count(*) FROM chunks'")
    [ "$chunks" = "134261" ] || fail "chunk count mismatch: got $chunks, expected 134261"
    ok "chunks: $chunks"

    docs=$(ssh "$DEST" "$RPATH PGPASSWORD=kp psql -h localhost -U kp -d knowledge_palace -t -A -c 'SELECT count(*) FROM documents'")
    ok "documents: $docs"

    log "  installing python env on target (uv sync)..."
    ssh "$DEST" "cd /Users/${DEST%%@*}/Documents/knowledge-palace && $RPATH uv sync 2>&1 | tail -2"
    ok "venv ready"

    log "  search smoke test on target (model downloads on first run, be patient)..."
    ssh "$DEST" "cd /Users/${DEST%%@*}/Documents/knowledge-palace && $RPATH uv run kp search 'LoRA finetuning' --mode keyword -n 2 2>&1 | grep -E 'Search:|Score' | head -4" \
        && ok "search works on target" || fail "search failed on target"
}

stage_books() {
    log "Stage: Calibre library ($(du -sh "$SRC_BOOKS" 2>/dev/null | cut -f1)) -> $DEST"
    rsync -az --info=progress2 \
        "$SRC_BOOKS/" "$DEST:/Users/${DEST%%@*}/Documents/Books/"
    ok "books synced"
}

case "$STAGE" in
    code)   stage_code ;;
    db)     stage_db ;;
    verify) stage_verify ;;
    books)  stage_books ;;
    all)    stage_code; stage_db; stage_verify; stage_books ;;
    *)      fail "unknown stage: $STAGE (use all|code|db|verify|books)" ;;
esac

log "Done: stage '$STAGE' complete for $DEST"
