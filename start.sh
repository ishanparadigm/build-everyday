#!/bin/bash
# start.sh — daily workflow command for build-everyday
#
# Usage:
#   ./start.sh              # start today's challenge (highest day folder)
#   ./start.sh 3            # start day 003 specifically
#   ./start.sh --reveal     # reveal today's reference solution
#   ./start.sh 3 --reveal   # reveal day 003's solution

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="/Users/ishan/build-everyday"
PROGRESS_FILE="$REPO_DIR/progress.json"
TIMER_DIR="/tmp/build-everyday"
mkdir -p "$TIMER_DIR"

# ── ANSI Colors ───────────────────────────────────────────────────────────────
BOLD='\033[1m'
DIM='\033[2m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'
BG_GREEN='\033[42m'
BG_RED='\033[41m'

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() {
  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║           build-everyday  //  start.sh          ║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
  echo ""
}

hr() {
  echo -e "${DIM}──────────────────────────────────────────────────${RESET}"
}

elapsed_str() {
  local secs=$1
  local mins=$(( secs / 60 ))
  local s=$(( secs % 60 ))
  if [ "$mins" -gt 0 ]; then
    printf "%dm %ds" "$mins" "$s"
  else
    printf "%ds" "$s"
  fi
}

# ── Parse Arguments ───────────────────────────────────────────────────────────
REVEAL=false
TARGET_DAY=""

for arg in "$@"; do
  case "$arg" in
    --reveal) REVEAL=true ;;
    [0-9]*) TARGET_DAY="$arg" ;;
    *)
      echo -e "${RED}Unknown argument: $arg${RESET}"
      echo "Usage: ./start.sh [DAY_NUMBER] [--reveal]"
      exit 1
      ;;
  esac
done

# ── Find the Target Day Folder ────────────────────────────────────────────────
cd "$REPO_DIR"

if [ -n "$TARGET_DAY" ]; then
  DAY_PADDED=$(printf '%03d' "$TARGET_DAY")
  DAY_FOLDER=$(find "$REPO_DIR" -maxdepth 1 -type d -name "day-${DAY_PADDED}-*" | head -1)
  if [ -z "$DAY_FOLDER" ]; then
    echo -e "${RED}Error: No folder found for day $DAY_PADDED${RESET}"
    exit 1
  fi
else
  # Find the highest-numbered day folder
  DAY_FOLDER=$(find "$REPO_DIR" -maxdepth 1 -type d -name 'day-*' | sort | tail -1)
  if [ -z "$DAY_FOLDER" ]; then
    echo -e "${RED}Error: No day-XXX folders found in $REPO_DIR${RESET}"
    exit 1
  fi
fi

FOLDER_NAME=$(basename "$DAY_FOLDER")
# Extract day number and topic from folder name like "day-003-logistic-regression"
DAY_NUM=$(echo "$FOLDER_NAME" | sed 's/day-\([0-9]*\)-.*/\1/')
DAY_INT=$((10#$DAY_NUM))
TOPIC_SLUG=$(echo "$FOLDER_NAME" | sed 's/day-[0-9]*-//')
TOPIC_DISPLAY=$(echo "$TOPIC_SLUG" | tr '-' ' ')

# Get the challenge title from progress.json
CHALLENGE_TITLE=$(python3 -c "
import json
with open('$PROGRESS_FILE') as f:
    data = json.load(f)
for d in data['days']:
    if d['day'] == $DAY_INT:
        print(d['challenge'])
        break
else:
    print('$TOPIC_DISPLAY')
" 2>/dev/null || echo "$TOPIC_DISPLAY")

# Timer file for this specific day
TIMER_FILE="$TIMER_DIR/day-${DAY_NUM}-start"

# ── Reveal Mode ───────────────────────────────────────────────────────────────
if [ "$REVEAL" = true ]; then
  SOLUTION_FILE="$DAY_FOLDER/solution.py"
  if [ ! -f "$SOLUTION_FILE" ]; then
    echo -e "${RED}Error: No solution.py found in $FOLDER_NAME${RESET}"
    exit 1
  fi

  banner
  echo -e "${YELLOW}${BOLD}  WARNING: You are about to reveal the reference solution${RESET}"
  echo -e "${YELLOW}  for Day $DAY_NUM: $CHALLENGE_TITLE${RESET}"
  echo ""
  echo -e "${DIM}  This will be recorded in progress.json.${RESET}"
  hr
  echo ""
  read -rp "$(echo -e "${YELLOW}  Are you sure? (y/N): ${RESET}")" confirm
  echo ""

  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    # Mark as revealed in progress.json
    python3 -c "
import json
with open('$PROGRESS_FILE', 'r') as f:
    data = json.load(f)
for d in data['days']:
    if d['day'] == $DAY_INT:
        d['revealed'] = True
        break
with open('$PROGRESS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
    echo -e "${MAGENTA}${BOLD}  ── Reference Solution: Day $DAY_NUM ──${RESET}"
    echo ""
    cat "$SOLUTION_FILE"
    echo ""
    hr
    echo -e "${DIM}  Revealed flag saved to progress.json.${RESET}"
  else
    echo -e "${GREEN}  Good call. Keep trying!${RESET}"
  fi
  exit 0
fi

# ── Banner & Day Info ─────────────────────────────────────────────────────────
banner
echo -e "  ${BOLD}Day $DAY_NUM${RESET}  ${CYAN}$CHALLENGE_TITLE${RESET}"
echo -e "  ${DIM}Folder: $FOLDER_NAME/${RESET}"
hr

# ── Check Required Files ─────────────────────────────────────────────────────
TESTS_FILE="$DAY_FOLDER/tests.py"
MY_SOLUTION="$DAY_FOLDER/my_solution.py"

if [ ! -f "$TESTS_FILE" ]; then
  echo -e "${RED}  Error: tests.py not found in $FOLDER_NAME${RESET}"
  exit 1
fi
if [ ! -f "$MY_SOLUTION" ]; then
  echo -e "${RED}  Error: my_solution.py not found in $FOLDER_NAME${RESET}"
  exit 1
fi

# ── Show Current Test Status ──────────────────────────────────────────────────
get_test_status() {
  # Run pytest and capture results; returns "passed/total" string
  local result
  result=$(cd "$DAY_FOLDER" && python3 -m pytest tests.py -v --tb=short 2>&1)
  local passed=$(echo "$result" | grep -cE "PASSED" || true)
  local failed=$(echo "$result" | grep -cE "FAILED|ERROR" || true)
  local total=$((passed + failed))
  echo "$passed $total"
  return 0
}

echo ""
echo -e "  ${BOLD}Current Status${RESET}"
STATUS=$(get_test_status)
PASSED=$(echo "$STATUS" | awk '{print $1}')
TOTAL=$(echo "$STATUS" | awk '{print $2}')

if [ "$TOTAL" -eq 0 ]; then
  echo -e "  ${YELLOW}No tests detected (or import errors). Check tests.py.${RESET}"
elif [ "$PASSED" -eq "$TOTAL" ]; then
  echo -e "  ${GREEN}${BOLD}$PASSED/$TOTAL tests passing${RESET}  ${GREEN}-- All clear!${RESET}"
else
  echo -e "  ${RED}$PASSED/$TOTAL tests passing${RESET}"
fi
hr

# ── Show README Teaser ────────────────────────────────────────────────────────
README_FILE="$DAY_FOLDER/README.md"
if [ -f "$README_FILE" ]; then
  echo ""
  echo -e "  ${BOLD}Challenge Description${RESET}  ${DIM}(first 30 lines)${RESET}"
  echo ""
  head -30 "$README_FILE" | sed 's/^/  /'
  TOTAL_LINES=$(wc -l < "$README_FILE" | tr -d ' ')
  if [ "$TOTAL_LINES" -gt 30 ]; then
    echo ""
    echo -e "  ${DIM}... ($((TOTAL_LINES - 30)) more lines — see $FOLDER_NAME/README.md)${RESET}"
  fi
  hr
fi

# ── Instructions ──────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Your task:${RESET} Edit ${CYAN}$FOLDER_NAME/my_solution.py${RESET}"
echo -e "  ${DIM}Tests run from: $FOLDER_NAME/tests.py${RESET}"
echo ""
echo -e "  ${BOLD}Watch mode:${RESET} Press ${WHITE}Enter${RESET} to re-run tests. Type ${WHITE}q${RESET} to quit."
hr

# ── Start Timer ───────────────────────────────────────────────────────────────
if [ -f "$TIMER_FILE" ]; then
  START_TIME=$(cat "$TIMER_FILE")
  NOW=$(date +%s)
  PREV_ELAPSED=$(( NOW - START_TIME ))
  echo ""
  echo -e "  ${DIM}Timer resumed — $(elapsed_str $PREV_ELAPSED) already elapsed.${RESET}"
else
  START_TIME=$(date +%s)
  echo "$START_TIME" > "$TIMER_FILE"
  echo ""
  echo -e "  ${DIM}Timer started.${RESET}"
fi
echo ""

# ── Watch Loop ────────────────────────────────────────────────────────────────
run_tests() {
  local now=$(date +%s)
  local start=$(cat "$TIMER_FILE")
  local elapsed=$(( now - start ))

  echo ""
  echo -e "${BLUE}${BOLD}  ── Test Run ──  ${RESET}${DIM}Elapsed: $(elapsed_str $elapsed)${RESET}"
  echo ""

  # Run pytest
  (cd "$DAY_FOLDER" && python3 -m pytest tests.py -v --tb=short 2>&1) | sed 's/^/  /'
  local pytest_exit=${PIPESTATUS[0]}

  echo ""

  # Parse results
  local result
  result=$(cd "$DAY_FOLDER" && python3 -m pytest tests.py --tb=no -q 2>&1)
  local passed=$(echo "$result" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
  local failed=$(echo "$result" | grep -oE '[0-9]+ (failed|error)' | grep -oE '[0-9]+' | head -1 || echo "0")
  passed=${passed:-0}
  failed=${failed:-0}
  local total=$((passed + failed))

  if [ "$total" -eq 0 ]; then
    echo -e "  ${YELLOW}Could not parse test results. Check for import errors.${RESET}"
    return 1
  fi

  if [ "$passed" -eq "$total" ] && [ "$total" -gt 0 ]; then
    echo -e "  ${BG_GREEN}${WHITE}${BOLD}  ALL $total TESTS PASSING  ${RESET}"
    echo ""
    echo -e "  ${GREEN}${BOLD}Congratulations!${RESET} You solved Day $DAY_NUM: $CHALLENGE_TITLE"
    echo -e "  ${GREEN}Time: $(elapsed_str $elapsed)${RESET}"
    hr
    return 0
  else
    echo -e "  ${RED}$PASSED/$TOTAL tests passing${RESET} ${DIM}— keep going!${RESET}"
    return 1
  fi
}

handle_completion() {
  local now=$(date +%s)
  local start=$(cat "$TIMER_FILE")
  local elapsed=$(( now - start ))
  local elapsed_display
  elapsed_display=$(elapsed_str $elapsed)
  local mins=$(( elapsed / 60 ))
  local secs=$(( elapsed % 60 ))
  local commit_time="${mins}m ${secs}s"

  echo ""
  read -rp "$(echo -e "${CYAN}  Commit your solution and push? (Y/n): ${RESET}")" do_commit
  echo ""

  if [[ ! "$do_commit" =~ ^[Nn]$ ]]; then
    cd "$REPO_DIR"

    # 1. Commit my_solution.py
    echo -e "  ${DIM}Staging my_solution.py...${RESET}"
    git add "$DAY_FOLDER/my_solution.py"
    git commit -m "Solve Day $DAY_NUM: $CHALLENGE_TITLE ($commit_time)"

    # 2. Update progress.json
    echo -e "  ${DIM}Updating progress.json...${RESET}"
    python3 -c "
import json
from datetime import date

with open('$PROGRESS_FILE', 'r') as f:
    data = json.load(f)

for d in data['days']:
    if d['day'] == $DAY_INT:
        d['my_completed'] = True
        d['my_time_seconds'] = $elapsed
        d['my_completed_date'] = date.today().isoformat()
        break

with open('$PROGRESS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"

    # 3. Regenerate README visuals
    echo -e "  ${DIM}Regenerating README visuals...${RESET}"
    python3 "$REPO_DIR/generate_readme.py" 2>&1 | sed 's/^/  /' || true

    # 4. Commit progress update
    git add "$PROGRESS_FILE" README.md docs/ 2>/dev/null || true
    git commit -m "Update progress: Day $DAY_NUM completed ($commit_time)" 2>/dev/null || true

    # 5. Push
    echo -e "  ${DIM}Pushing to origin main...${RESET}"
    if git push origin main 2>&1 | sed 's/^/  /'; then
      echo ""
      echo -e "  ${GREEN}${BOLD}Pushed successfully!${RESET}"
    else
      echo ""
      echo -e "  ${YELLOW}Push failed — you can push manually later.${RESET}"
    fi

    # Clean up timer
    rm -f "$TIMER_FILE"

    echo ""
    hr
    echo -e "  ${GREEN}${BOLD}Day $DAY_NUM complete. See you tomorrow!${RESET}"
    hr
    echo ""
  else
    echo -e "  ${DIM}Skipped. You can commit manually whenever you're ready.${RESET}"
    rm -f "$TIMER_FILE"
  fi
}

# Main watch loop
while true; do
  if run_tests; then
    handle_completion
    exit 0
  fi

  echo ""
  read -rp "$(echo -e "${DIM}  Press Enter to re-run tests (q to quit): ${RESET}")" input
  if [[ "$input" =~ ^[Qq]$ ]]; then
    local_now=$(date +%s)
    local_start=$(cat "$TIMER_FILE")
    local_elapsed=$(( local_now - local_start ))
    echo ""
    echo -e "  ${DIM}Session paused at $(elapsed_str $local_elapsed). Timer saved — it will resume next time.${RESET}"
    echo ""
    exit 0
  fi
done
