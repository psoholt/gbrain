#!/usr/bin/env bash
# tests/heavy/sync_lock_regression.sh
# Sync writer-lock concurrency regression test.
#
# Asserts the cross-process sync writer-lock contract:
#   1. Exclusion + fail-fast: while the lock is held, N concurrent
#      `gbrain sync` processes ALL fail fast with "Another sync is in
#      progress" — none acquire, none queue/hang.
#   2. Acquire-when-free: once the lock is released, a single sync acquires
#      it, does its work, and completes.
#   3. No leak: after every sync exits, zero `gbrain_cycle_locks` rows for
#      the sync lock id remain (release via try/finally).
#
# Lock id: the single-default-source CLI path (`gbrain sync --dir`, no
# --source) takes the bare writer lock keyed `syncLockId('default')` =
# 'gbrain-sync:default' (src/core/db-lock.ts + src/commands/sync.ts
# performSync). v0.40.5.0 namespaced the lock per source; before that it was
# the bare 'gbrain-sync' constant.
#
# Why we HOLD the lock instead of racing N processes for it: the original
# "spawn N, expect exactly 1 winner" shape was timing-fragile. On a 2-page
# brain the winner's sync finishes in well under the inter-process startup
# stagger, so it releases the lock before the other processes reach the
# acquire — they then acquire serially and you see 2+ "winners". That flaked
# CI (got 2 winners, expected 1). Holding the lock externally for the whole
# spawn window removes the race: exclusion is asserted deterministically.
#
# Postgres-only (no DATABASE_URL = graceful skip with hint).

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -z "${DATABASE_URL:-}" ]; then
  echo "[sync_lock_regression] DATABASE_URL not set; skipping (informational)." >&2
  echo "  Local: docker run -d --name gbrain-test-pg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=gbrain_test -p 5434:5432 pgvector/pgvector:pg16" >&2
  echo "  Then: export DATABASE_URL=postgresql://postgres:postgres@localhost:5434/gbrain_test" >&2
  exit 0
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "[sync_lock_regression] psql required. Install postgresql-client." >&2
  exit 2
fi

TS=$(date -u +%Y%m%d-%H%M%SZ)
# Isolate from the developer's real ~/.gbrain so writing sync.repo_path doesn't
# clobber their config. Restored on exit.
TMP_GBRAIN_HOME=$(mktemp -d -t gbrain-sync-lock-home-XXXXXX)
export GBRAIN_HOME="$TMP_GBRAIN_HOME"
LOG_DIR="$GBRAIN_HOME/audit"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/heavy-sync_lock_regression-$TS.log"
# Surface the log path so it survives the EXIT trap that nukes GBRAIN_HOME.
SURFACE_LOG="${TMPDIR:-/tmp}/heavy-sync_lock_regression-$TS.log"
trap 'rm -rf "$TMP_GBRAIN_HOME"; cp -f "$LOG" "$SURFACE_LOG" 2>/dev/null || true' EXIT

NUM_PARALLEL="${NUM_PARALLEL:-4}"
echo "[sync_lock_regression] DATABASE_URL=$DATABASE_URL"
echo "[sync_lock_regression] log=$LOG"
echo "[sync_lock_regression] spawning $NUM_PARALLEL parallel sync processes..."

# Step 1: ensure schema is up-to-date by running doctor once. Doctor exits
# non-zero when ANY check warns (e.g. missing embedding provider on a fresh
# CI runner) so we ignore its exit status — the schema-migration side effect
# is what we want here, and the migration runs regardless of check verdicts.
echo "[sync_lock_regression] init schema via gbrain doctor..." | tee -a "$LOG"
timeout 180s bun run src/cli.ts doctor --json > /dev/null 2>>"$LOG" || true

# Step 2: create a tiny brain dir + register it as sync.repo_path so each sync
# call has something legitimate to do.
BRAIN_DIR=$(mktemp -d -t gbrain-sync-lock-XXXXXX)
# Compose with the earlier GBRAIN_HOME-cleanup trap (NOT overwrite it).
trap 'psql "$DATABASE_URL" -c "DELETE FROM gbrain_cycle_locks WHERE id='"'"'gbrain-sync:default'"'"' AND holder_pid=999999" >/dev/null 2>&1 || true; rm -rf "$BRAIN_DIR" "$TMP_GBRAIN_HOME"; cp -f "$LOG" "$SURFACE_LOG" 2>/dev/null || true' EXIT

# Seed two markdown pages so sync has real (but trivial) work
mkdir -p "$BRAIN_DIR"
cat > "$BRAIN_DIR/page-a.md" <<'EOF'
---
title: Lock Test Page A
---
# Lock Test Page A
Trivial content for sync-lock-regression heavy test.
EOF
cat > "$BRAIN_DIR/page-b.md" <<'EOF'
---
title: Lock Test Page B
---
# Lock Test Page B
Trivial content for sync-lock-regression heavy test.
EOF

# git-init so sync's diff-walk has something to anchor (sync expects a git repo)
(cd "$BRAIN_DIR" && git init -q && git add . && git -c user.email=test@test -c user.name=test commit -q -m "seed" >/dev/null 2>&1) || true

# Tell gbrain to use this brain dir. v0.41 introduced the source registry
# (sources table) as the canonical "where do pages come from" surface;
# `sync.repo_path` is the legacy key and sync now reads the source row's
# `local_path` column. Update the default source's local_path directly via
# psql (mirrors how fm_wallclock.sh registers via the engine API — same
# semantics, lower process-spawn overhead).
psql "$DATABASE_URL" -c "INSERT INTO sources (id, name, local_path) VALUES ('default', 'default', '$BRAIN_DIR') ON CONFLICT (id) DO UPDATE SET local_path = EXCLUDED.local_path;" >>"$LOG" 2>&1
# Keep the legacy config key set too — some code paths still read it, and
# setting both is the belt-and-suspenders shape downstream callers expect.
bun run src/cli.ts config set sync.repo_path "$BRAIN_DIR" >/dev/null 2>&1 || true

# The single-default-source CLI path takes this lock id (see header).
LOCK_ID="gbrain-sync:default"
SENTINEL_PID=999999

# Step 3 (Part A — exclusion + fail-fast): hold the sync lock externally with
# a far-future TTL, then spawn N concurrent syncs. With the lock held they
# must ALL fail fast with "Another sync is in progress" — deterministic, no
# race for who-wins.
echo "[sync_lock_regression] holding $LOCK_ID externally (sentinel pid $SENTINEL_PID)..." | tee -a "$LOG"
psql "$DATABASE_URL" -t -A -c "INSERT INTO gbrain_cycle_locks (id, holder_pid, holder_host, acquired_at, ttl_expires_at) VALUES ('$LOCK_ID', $SENTINEL_PID, 'sync-lock-regression-sentinel', NOW(), NOW() + interval '10 minutes') ON CONFLICT (id) DO UPDATE SET holder_pid = $SENTINEL_PID, holder_host = 'sync-lock-regression-sentinel', acquired_at = NOW(), ttl_expires_at = NOW() + interval '10 minutes';" >>"$LOG" 2>&1

# Spawn N parallel sync processes. Capture each one's exit code + stdout/stderr.
PIDS=()
EXIT_FILES=()
OUT_FILES=()
for ((i=1; i<=NUM_PARALLEL; i+=1)); do
  EXIT_F=$(mktemp -t sync-lock-exit-XXXXXX)
  OUT_F=$(mktemp -t sync-lock-out-XXXXXX)
  EXIT_FILES+=("$EXIT_F")
  OUT_FILES+=("$OUT_F")
  # --no-embed: this test measures the writer-lock race, not embeddings.
  # CI runners don't pipe ZEROENTROPY_API_KEY / OPENAI_API_KEY / VOYAGE_API_KEY,
  # so without --no-embed every sync fails with "Embedding model X requires Y"
  # and the test classifier reports unknown failures instead of lock outcomes.
  # --repo: sync's canonical brain-dir flag (the older --dir is silently
  # ignored; the script previously paired it with `config set sync.repo_path`
  # which sync no longer reads in the source-registry world).
  ( bun run src/cli.ts sync --repo "$BRAIN_DIR" --no-embed >"$OUT_F" 2>&1; echo $? > "$EXIT_F" ) &
  PIDS+=($!)
done

echo "[sync_lock_regression] waiting on ${#PIDS[@]} pids..."
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

# Step 4: collect outcomes
WINNERS=0
LOSERS=0
UNKNOWN=0
for ((i=0; i<NUM_PARALLEL; i+=1)); do
  rc=$(cat "${EXIT_FILES[$i]}" 2>/dev/null || echo "?")
  out_file="${OUT_FILES[$i]}"
  if [ "$rc" = "0" ]; then
    WINNERS=$((WINNERS + 1))
    echo "  [sync $((i+1))] rc=0 (winner)" | tee -a "$LOG"
  elif grep -q "Another sync is in progress" "$out_file" 2>/dev/null; then
    LOSERS=$((LOSERS + 1))
    echo "  [sync $((i+1))] rc=$rc (lock-busy: 'Another sync is in progress')" | tee -a "$LOG"
  else
    UNKNOWN=$((UNKNOWN + 1))
    echo "  [sync $((i+1))] rc=$rc (unknown failure — see $out_file)" | tee -a "$LOG"
    head -5 "$out_file" 2>/dev/null | sed 's/^/    > /' | tee -a "$LOG"
  fi
done

# Cleanup tmp files
rm -f "${EXIT_FILES[@]}" "${OUT_FILES[@]}"

echo "[sync_lock_regression] outcomes (lock held): winners=$WINNERS losers=$LOSERS unknown=$UNKNOWN" | tee -a "$LOG"

# Release the externally-held lock so a free sync can acquire it.
echo "[sync_lock_regression] releasing externally-held $LOCK_ID..." | tee -a "$LOG"
psql "$DATABASE_URL" -t -A -c "DELETE FROM gbrain_cycle_locks WHERE id = '$LOCK_ID' AND holder_pid = $SENTINEL_PID;" >>"$LOG" 2>&1

# Step 4 (Part B — acquire-when-free + release): with the lock free, a single
# sync must acquire it, do its work, and complete (rc=0). Proves the lock
# isn't stuck-held and a legitimate sync still works.
FREE_OUT=$(mktemp -t sync-lock-free-XXXXXX)
# Match Part A's flags. --repo is sync's canonical brain-dir flag (the older
# --dir is silently ignored by sync's arg parser); --no-embed dodges the
# embed-creds preflight (src/commands/sync.ts) that process.exit(1)s on CI
# runners which don't pipe an embedding API key. Without both, this free sync
# exits 1, and under `set -e` a bare `( ... ); FREE_RC=$?` aborts the script
# before FREE_RC is captured — the Part B verdict never runs. Capture via
# `|| FREE_RC=$?` so a nonzero sync is reported by the verdict below instead.
FREE_RC=0
( bun run src/cli.ts sync --repo "$BRAIN_DIR" --no-embed >"$FREE_OUT" 2>&1 ) || FREE_RC=$?
if [ "$FREE_RC" = "0" ]; then
  echo "[sync_lock_regression] free-lock sync: rc=0 (acquired + completed)" | tee -a "$LOG"
else
  echo "[sync_lock_regression] free-lock sync: rc=$FREE_RC (FAILED — see below)" | tee -a "$LOG"
  head -5 "$FREE_OUT" 2>/dev/null | sed 's/^/    > /' | tee -a "$LOG"
fi
rm -f "$FREE_OUT"

# Step 5: assert no leaked gbrain_cycle_locks rows for the sync lock id. The
# pkey column is `id`, not `lock_id` (confirmed via \d gbrain_cycle_locks).
LEAKED=$(psql "$DATABASE_URL" -t -A -c "SELECT COUNT(*) FROM gbrain_cycle_locks WHERE id = '$LOCK_ID';" 2>>"$LOG" | tr -d ' ')
echo "[sync_lock_regression] post-run gbrain_cycle_locks($LOCK_ID) row count: $LEAKED" | tee -a "$LOG"

# Step 6: verdict
FAIL=0

# Part A: while the lock was held, EVERY spawned sync must have failed fast
# with the lock-busy message. A winner means mutual exclusion was breached
# (a second sync acquired a lock another holder already had).
if [ "$WINNERS" -ne 0 ]; then
  echo "[sync_lock_regression] FAIL: lock was held — expected 0 winners, got $WINNERS (mutual exclusion breached)" >&2
  FAIL=1
fi

# All N must be lock-busy losers — anything else (an unknown failure) means a
# sync exited for a reason other than the lock, which would taint the result.
if [ "$LOSERS" -ne "$NUM_PARALLEL" ]; then
  echo "[sync_lock_regression] FAIL: expected $NUM_PARALLEL lock-busy losers, got $LOSERS (unknown failures: $UNKNOWN)" >&2
  FAIL=1
fi

# Part B: with the lock free, the single sync must have acquired and completed.
if [ "$FREE_RC" != "0" ]; then
  echo "[sync_lock_regression] FAIL: free-lock sync did not succeed (rc=$FREE_RC) — lock stuck-held or sync broken" >&2
  FAIL=1
fi

# The lock row must be cleaned up on exit (release via try/finally).
if [ "$LEAKED" != "0" ]; then
  echo "[sync_lock_regression] FAIL: $LEAKED leaked gbrain_cycle_locks($LOCK_ID) row(s) after all syncs exited" >&2
  FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  echo "[sync_lock_regression] FAILED. See $LOG for details." >&2
  exit 1
fi

echo "[sync_lock_regression] OK — held lock blocked all $NUM_PARALLEL syncs (fail-fast), free-lock sync succeeded, no leaked lock rows."
