#!/bin/bash
# daily-build.sh — generates daily coding challenges via Claude CLI
#
# Features:
#   - Catches up missed days automatically (capped at 5)
#   - Backdates git commits for missed days so GitHub contribution graph is correct
#   - Locks against concurrent runs
#   - Rotates logs
#   - Smoke-tests generated solutions (python3 solution.py)
#   - Validates track assignment matches the expected day-of-week
#   - Sends macOS notification on failure
#   - Loads prompt from prompt.md (not inline)

set -euo pipefail

REPO_DIR="/Users/ishan/build-everyday"
LOG_FILE="$REPO_DIR/cron.log"
LOCK_FILE="$REPO_DIR/.daily-build.lock"
PROMPT_FILE="$REPO_DIR/prompt.md"
PROGRESS_FILE="$REPO_DIR/progress.json"
CURRICULUM_FILE="$REPO_DIR/curriculum.json"
MAX_LOG_LINES=500
MAX_CATCHUP=5
START_DATE="2026-04-01"

# --- PATH setup (resilient to nvm version changes) ---
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  export NVM_DIR="$HOME/.nvm"
  # shellcheck disable=SC1091
  source "$NVM_DIR/nvm.sh" --no-use
  nvm use default --silent 2>/dev/null || true
fi
if ! command -v node &>/dev/null; then
  NODE_PATH=$(find "$HOME/.nvm/versions/node" -name node -type f 2>/dev/null | sort -V | tail -1)
  [ -n "${NODE_PATH:-}" ] && export PATH="$(dirname "$NODE_PATH"):$PATH"
fi

# --- Helpers ---
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

notify_failure() {
  local msg="$1"
  log "NOTIFY: $msg"
  # macOS notification
  if command -v osascript &>/dev/null; then
    osascript -e "display notification \"$msg\" with title \"build-everyday\" subtitle \"Daily Build Failed\" sound name \"Basso\"" 2>/dev/null || true
  fi
}

# Map day-of-week number (1=Mon..7=Sun) to expected track
expected_track_for_dow() {
  case "$1" in
    1|4) echo "ai" ;;
    2|5) echo "crypto" ;;
    3|6) echo "robotics" ;;
    7)   echo "integration" ;;
    *)   echo "unknown" ;;
  esac
}

# --- Lock to prevent concurrent runs ---
if [ -f "$LOCK_FILE" ]; then
  LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
  if kill -0 "$LOCK_PID" 2>/dev/null; then
    log "SKIP: another instance running (PID $LOCK_PID)"
    exit 0
  else
    log "WARN: stale lock from PID $LOCK_PID, removing"
    rm -f "$LOCK_FILE"
  fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# --- Log rotation ---
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt "$MAX_LOG_LINES" ]; then
  tail -n "$MAX_LOG_LINES" "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
  log "LOG: rotated to last $MAX_LOG_LINES lines"
fi

cd "$REPO_DIR"

# --- Validate prompt file exists ---
if [ ! -f "$PROMPT_FILE" ]; then
  notify_failure "prompt.md not found"
  exit 1
fi

# --- Pull latest to avoid conflicts ---
git pull --rebase --quiet origin main 2>/dev/null || log "WARN: git pull failed, continuing with local state"

# --- Determine how many days to generate ---
current_day_count=$(find "$REPO_DIR" -maxdepth 1 -type d -name 'day-*' | wc -l | tr -d ' ')

today=$(date '+%Y-%m-%d')
start_epoch=$(date -j -f '%Y-%m-%d' "$START_DATE" '+%s')
today_epoch=$(date -j -f '%Y-%m-%d' "$today" '+%s')
expected_days=$(( (today_epoch - start_epoch) / 86400 ))

days_behind=$(( expected_days - current_day_count ))

if [ "$days_behind" -le 0 ]; then
  log "OK: up to date ($current_day_count days done, $expected_days expected)"
  exit 0
fi

if [ "$days_behind" -gt "$MAX_CATCHUP" ]; then
  log "WARN: $days_behind days behind, capping catch-up to $MAX_CATCHUP"
  days_behind=$MAX_CATCHUP
fi

log "START: $current_day_count days exist, $expected_days expected, generating $days_behind"

# --- Generate each missing day ---
failures=0
for i in $(seq 1 "$days_behind"); do
  next_day=$(( current_day_count + i ))
  next_day_padded=$(printf '%03d' "$next_day")

  # Calculate the actual date this day should have been generated
  day_offset=$(( current_day_count + i ))
  target_epoch=$(( start_epoch + day_offset * 86400 ))
  target_date=$(date -j -f '%s' "$target_epoch" '+%Y-%m-%d')
  target_dow=$(date -j -f '%s' "$target_epoch" '+%u')  # 1=Mon..7=Sun
  expected_track=$(expected_track_for_dow "$target_dow")

  log "GENERATING: Day $next_day_padded (target date: $target_date, track: $expected_track, $i of $days_behind)"

  # Run Claude with the external prompt
  PROMPT_CONTENT=$(cat "$PROMPT_FILE")
  if /opt/homebrew/bin/claude --print --dangerously-skip-permissions "$PROMPT_CONTENT" \
    >> "$LOG_FILE" 2>&1; then

    # --- Smoke test: find the newly created folder and run solution.py ---
    new_folder=$(find "$REPO_DIR" -maxdepth 1 -type d -name "day-${next_day_padded}-*" | head -1)
    if [ -n "$new_folder" ] && [ -f "$new_folder/solution.py" ]; then
      log "SMOKE TEST: running $new_folder/solution.py"
      if python3 "$new_folder/solution.py" > /dev/null 2>&1; then
        log "SMOKE TEST: passed"
      else
        log "WARN: smoke test failed for $new_folder/solution.py (exit code $?), keeping anyway"
      fi
    fi

    # --- Track validation: check folder name matches expected track ---
    if [ -n "$new_folder" ]; then
      folder_name=$(basename "$new_folder")
      # Read the track from progress.json (last entry)
      if command -v python3 &>/dev/null && [ -f "$PROGRESS_FILE" ]; then
        actual_track=$(python3 -c "
import json
with open('$PROGRESS_FILE') as f:
    d = json.load(f)
if d['days']:
    print(d['days'][-1].get('track', 'unknown'))
else:
    print('unknown')
" 2>/dev/null || echo "unknown")
        if [ "$actual_track" != "$expected_track" ]; then
          log "WARN: track mismatch — expected '$expected_track' (${target_date}), got '$actual_track' in $folder_name"
        else
          log "TRACK: verified $actual_track matches expected for $target_date"
        fi
      fi
    fi

    # --- Backdate the commit if this is a catch-up day ---
    if [ "$target_date" != "$today" ]; then
      # Amend the last commit with the correct date
      BACKDATE="${target_date}T12:00:00"
      GIT_AUTHOR_DATE="$BACKDATE" GIT_COMMITTER_DATE="$BACKDATE" \
        git commit --amend --no-edit --date="$BACKDATE" --allow-empty >> "$LOG_FILE" 2>&1 || true
      log "BACKDATE: amended commit to $target_date"
    fi

    log "DONE: Day $next_day_padded generated and pushed"
  else
    exit_code=$?
    log "ERROR: Day $next_day_padded failed (exit code $exit_code)"
    notify_failure "Day $next_day_padded generation failed (exit $exit_code)"
    failures=$((failures + 1))
  fi
done

# --- Final status ---
if [ "$failures" -gt 0 ]; then
  notify_failure "$failures of $days_behind days failed"
  log "FINISHED with $failures/$days_behind failures"
  exit 1
else
  log "FINISHED: all $days_behind days generated successfully"
fi
