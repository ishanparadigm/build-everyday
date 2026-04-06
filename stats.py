#!/usr/bin/env python3
"""build-everyday terminal dashboard — beautiful stats at a glance."""

import calendar
import json
import os
import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(REPO_DIR, "progress.json")
CURRICULUM_FILE = os.path.join(REPO_DIR, "curriculum.json")
START_DATE = datetime(2026, 4, 1)

# ── ANSI Colors ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Track colors
BLUE = "\033[38;5;75m"       # AI
YELLOW = "\033[38;5;214m"    # Crypto
GREEN = "\033[38;5;114m"     # Robotics
MAGENTA = "\033[38;5;176m"   # Integration
WHITE = "\033[38;5;255m"
GRAY = "\033[38;5;242m"
BRIGHT_WHITE = "\033[38;5;15m"

# Background variants for calendar
BG_BLUE = "\033[48;5;75m\033[38;5;0m"
BG_YELLOW = "\033[48;5;214m\033[38;5;0m"
BG_GREEN = "\033[48;5;114m\033[38;5;0m"
BG_MAGENTA = "\033[48;5;176m\033[38;5;0m"
BG_GRAY = "\033[48;5;236m\033[38;5;245m"

TRACK_COLOR = {
    "ai": BLUE,
    "crypto": YELLOW,
    "robotics": GREEN,
    "integration": MAGENTA,
}

TRACK_BG = {
    "ai": BG_BLUE,
    "crypto": BG_YELLOW,
    "robotics": BG_GREEN,
    "integration": BG_MAGENTA,
}

TRACK_LABEL = {
    "ai": "AI",
    "crypto": "Crypto",
    "robotics": "Robotics",
    "integration": "Integrate",
}

TRACK_TOTALS = {
    "ai": 24,
    "crypto": 24,
    "robotics": 24,
    "integration": 12,
}

DOW_TO_TRACK = {
    0: "ai",        # Monday
    1: "crypto",     # Tuesday
    2: "robotics",   # Wednesday
    3: "ai",         # Thursday
    4: "crypto",     # Friday
    5: "robotics",   # Saturday
    6: "integration", # Sunday
}

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ── Box Drawing ──────────────────────────────────────────────────────────────

W = 54  # inner width (between the vertical bars)

def box_top():
    return f"  {GRAY}\u2554{'═' * W}\u2557{RESET}"

def box_mid():
    return f"  {GRAY}\u2560{'═' * W}\u2563{RESET}"

def box_bot():
    return f"  {GRAY}\u255a{'═' * W}\u255d{RESET}"

def box_row(content, raw_len=None):
    """Pad content to fill the box width. raw_len = visible char count (without ANSI)."""
    if raw_len is None:
        # Strip ANSI to count visible chars
        raw_len = len(re.sub(r'\033\[[0-9;]*m', '', content))
    pad = W - raw_len
    if pad < 0:
        pad = 0
    return f"  {GRAY}\u2551{RESET} {content}{' ' * (pad - 1)}{GRAY}\u2551{RESET}"

def box_empty():
    return box_row("", 0)

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)

def load_curriculum():
    with open(CURRICULUM_FILE) as f:
        return json.load(f)

# ── Streak Calculation ───────────────────────────────────────────────────────

def compute_streaks(days):
    """Compute current streak and longest streak based on calendar dates with activity."""
    if not days:
        return 0, 0

    # Get unique dates with activity, sorted
    dates = sorted(set(datetime.strptime(d["date"], "%Y-%m-%d").date() for d in days))

    # Longest streak of consecutive dates
    longest = 1
    current = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            current += 1
            longest = max(longest, current)
        elif dates[i] != dates[i - 1]:
            current = 1

    # Current streak: count back from the most recent date
    today = datetime.now().date()
    # If the most recent activity is not today or yesterday, streak is over
    if (today - dates[-1]).days > 1:
        cur_streak = 0
    else:
        cur_streak = 1
        for i in range(len(dates) - 2, -1, -1):
            if (dates[i + 1] - dates[i]).days == 1:
                cur_streak += 1
            elif dates[i + 1] != dates[i]:
                break

    return cur_streak, longest

# ── Progress Bar ─────────────────────────────────────────────────────────────

def progress_bar(done, total, width=20, color=""):
    filled = int(width * done / total) if total > 0 else 0
    empty = width - filled
    bar = f"{color}{'█' * filled}{GRAY}{'░' * empty}{RESET}"
    return bar

# ── Calendar ─────────────────────────────────────────────────────────────────

def render_calendar(days):
    """Render a monthly calendar with colored day squares."""
    today = datetime.now().date()
    year, month = today.year, today.month

    # Build map: date -> track (use first track if multiple challenges on same day)
    date_track = {}
    for d in days:
        dt = datetime.strptime(d["date"], "%Y-%m-%d").date()
        if dt not in date_track:
            date_track[dt] = d["track"]

    # Month info
    month_name = today.strftime("%b %Y")
    cal = calendar.monthcalendar(year, month)

    lines = []
    header = f"  {BRIGHT_WHITE}{month_name}{RESET}"
    lines.append((header, len(f"  {month_name}")))
    dow_header = f"  {DIM}Mo Tu We Th Fr Sa Su{RESET}"
    lines.append((dow_header, len("  Mo Tu We Th Fr Sa Su")))

    for week in cal:
        row = "  "
        row_raw = "  "
        for i, day_num in enumerate(week):
            if day_num == 0:
                row += "   "
                row_raw += "   "
            else:
                d = datetime(year, month, day_num).date()
                day_str = f"{day_num:2d}"
                if d in date_track:
                    bg = TRACK_BG.get(date_track[d], BG_GRAY)
                    row += f"{bg}{day_str}{RESET} "
                elif d == today:
                    row += f"{BOLD}{WHITE}{day_str}{RESET} "
                elif d > today:
                    row += f"{GRAY}{day_str}{RESET} "
                else:
                    # Past day with no activity
                    row += f"{DIM}{day_str}{RESET} "
                row_raw += f"{day_str} "
        lines.append((row, len(row_raw)))

    return lines

# ── Tier Info ────────────────────────────────────────────────────────────────

def get_tier(day_count):
    week = ((day_count - 1) // 7) + 1 if day_count > 0 else 1
    if week <= 4:
        return "Weeks 1-4 (Fundamentals)"
    elif week <= 8:
        return "Weeks 5-8 (Applied)"
    else:
        return "Weeks 9-12 (Advanced)"

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    progress = load_progress()
    days = progress["days"]

    total_days = len(days)
    current_streak, longest_streak = compute_streaks(days)

    # Track counts
    track_counts = defaultdict(int)
    for d in days:
        track_counts[d["track"]] += 1

    # Next day info
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    tomorrow_dow = tomorrow.weekday()  # 0=Mon
    next_track = DOW_TO_TRACK[tomorrow_dow]
    next_day_name = DOW_NAMES[tomorrow_dow]

    # ── Render ───────────────────────────────────────────────────────────
    out = []
    out.append("")
    out.append(box_top())

    # Title
    title = f"build-everyday  \u00b7  Day {total_days}"
    title_raw_len = len(title)
    pad_left = (W - 2 - title_raw_len) // 2
    pad_right = W - 2 - title_raw_len - pad_left
    title_line = f"{' ' * pad_left}{BOLD}{BRIGHT_WHITE}{title}{RESET}{' ' * pad_right}"
    out.append(box_row(title_line, W - 1))

    out.append(box_mid())

    # Streak
    streak_text = f"  Current Streak: {BOLD}{WHITE}{current_streak} days{RESET}    Longest: {BOLD}{WHITE}{longest_streak} days{RESET}"
    streak_raw = f"  Current Streak: {current_streak} days    Longest: {longest_streak} days"
    out.append(box_row(streak_text, len(streak_raw)))

    out.append(box_mid())

    # Track progress
    out.append(box_row(f"  {BOLD}{BRIGHT_WHITE}TRACK PROGRESS{RESET}", len("  TRACK PROGRESS")))
    out.append(box_empty())

    for track in ["ai", "crypto", "robotics", "integration"]:
        label = TRACK_LABEL[track]
        color = TRACK_COLOR[track]
        done = track_counts.get(track, 0)
        total = TRACK_TOTALS[track]
        pct = int(100 * done / total) if total > 0 else 0
        bar = progress_bar(done, total, width=22, color=color)

        # e.g. "  AI        ████░░░░  3/24  13%"
        label_padded = f"{label:<10s}"
        count_str = f"{done}/{total}"
        pct_str = f"{pct:>3d}%"

        content = f"  {color}{label_padded}{RESET}{bar}  {WHITE}{count_str:>5s}  {pct_str}{RESET}"
        raw = f"  {label_padded}" + ("█" * 22) + f"  {count_str:>5s}  {pct_str}"
        out.append(box_row(content, len(raw)))

    out.append(box_mid())

    # Calendar
    out.append(box_row(f"  {BOLD}{BRIGHT_WHITE}CALENDAR{RESET}", len("  CALENDAR")))
    cal_lines = render_calendar(days)
    for content, raw_len in cal_lines:
        out.append(box_row(content, raw_len))

    out.append(box_mid())

    # Recent
    out.append(box_row(f"  {BOLD}{BRIGHT_WHITE}RECENT{RESET}", len("  RECENT")))
    out.append(box_empty())

    recent = list(reversed(days[-5:]))
    for d in recent:
        day_num = f"{d['day']:03d}"
        track = d["track"]
        color = TRACK_COLOR.get(track, WHITE)
        track_label = TRACK_LABEL.get(track, track).upper()
        challenge = d["challenge"]
        # Truncate challenge if too long
        max_chal = 26
        if len(challenge) > max_chal:
            challenge = challenge[:max_chal - 1] + "\u2026"
        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        date_str = dt.strftime("%b %d")

        content = f"  {DIM}{day_num}{RESET}  {color}{track_label:<8s}{RESET}  {WHITE}{challenge:<{max_chal}s}{RESET}  {DIM}{date_str}{RESET}"
        raw = f"  {day_num}  {track_label:<8s}  {challenge:<{max_chal}s}  {date_str}"
        out.append(box_row(content, len(raw)))

    out.append(box_mid())

    # Next up
    out.append(box_row(f"  {BOLD}{BRIGHT_WHITE}NEXT UP{RESET}", len("  NEXT UP")))
    out.append(box_empty())

    next_color = TRACK_COLOR.get(next_track, WHITE)
    next_label = TRACK_LABEL.get(next_track, next_track)
    tomorrow_str = f"Tomorrow: {next_label} ({next_day_name})"
    out.append(box_row(f"  {next_color}{tomorrow_str}{RESET}", len(f"  {tomorrow_str}")))

    tier = get_tier(total_days)
    tier_str = f"Tier: {tier}"
    out.append(box_row(f"  {DIM}{tier_str}{RESET}", len(f"  {tier_str}")))

    out.append(box_empty())
    out.append(box_bot())
    out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
