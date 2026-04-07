#!/usr/bin/env python3
"""
generate_readme.py — Regenerate visual sections of README.md from progress.json and curriculum.json.

Generates:
  - docs/heatmap.svg (GitHub-contribution-calendar style)
  - Shields.io badge row
  - Streak stats
  - Track progress bars
  - Updates README.md in place using marker comments

Idempotent: safe to run multiple times.
Uses only the Python standard library.
"""

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent
PROGRESS_JSON = REPO / "progress.json"
CURRICULUM_JSON = REPO / "curriculum.json"
README_PATH = REPO / "README.md"
DOCS_DIR = REPO / "docs"
HEATMAP_PATH = DOCS_DIR / "heatmap.svg"

# ── Colours ──────────────────────────────────────────────────────────────────

TRACK_COLORS = {
    "ai": "#4A90D9",
    "crypto": "#E8873D",
    "robotics": "#50C878",
    "integration": "#9B59B6",
}
COLOR_EMPTY = "#2D2D2D"
COLOR_BG = "#161B22"
COLOR_TEXT = "#8B949E"
COLOR_TEXT_BRIGHT = "#C9D1D9"

START_DATE = date(2026, 4, 1)  # Wednesday

# ── Data loading ─────────────────────────────────────────────────────────────


def load_progress():
    with open(PROGRESS_JSON) as f:
        return json.load(f)


def load_curriculum():
    with open(CURRICULUM_JSON) as f:
        return json.load(f)


# ── Streak calculation ───────────────────────────────────────────────────────


def compute_streaks(progress):
    """Return (current_streak, longest_streak) based on calendar dates."""
    if not progress["days"]:
        return 0, 0

    # Include both generation dates and solve dates as "active" days
    raw_dates = set()
    for d in progress["days"]:
        raw_dates.add(d["date"])
        if d.get("my_completed_date"):
            raw_dates.add(d["my_completed_date"])
    date_set = {datetime.strptime(d, "%Y-%m-%d").date() for d in raw_dates}

    # Current streak: count backwards from today
    today = date.today()
    current = 0
    d = today
    while d in date_set:
        current += 1
        d -= timedelta(days=1)

    # Longest streak
    sorted_dates = sorted(date_set)
    longest = 0
    run = 1
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] - sorted_dates[i - 1] == timedelta(days=1):
            run += 1
        else:
            longest = max(longest, run)
            run = 1
    longest = max(longest, run)

    return current, longest


# ── Track progress ───────────────────────────────────────────────────────────


def compute_track_progress(progress, curriculum):
    """Return dict {track: (completed, total)} using case-insensitive matching."""
    tracks = curriculum["tracks"]
    result = {}
    for track_name, track_data in tracks.items():
        total = sum(len(v) for v in track_data["weeks"].values())
        # Completed challenges for this track (case-insensitive)
        completed_names = {
            d["challenge"].lower()
            for d in progress["days"]
            if d["track"] == track_name
        }
        all_challenges = []
        for tier_challenges in track_data["weeks"].values():
            all_challenges.extend(c.lower() for c in tier_challenges)
        completed = len(completed_names & set(all_challenges))
        result[track_name] = (completed, total)
    return result


# ── LOC stats ────────────────────────────────────────────────────────────────


def compute_loc(progress):
    """Count lines in each day folder's solution.py. Return (total_loc, per_day list)."""
    total = 0
    per_day = []
    for entry in progress["days"]:
        sol = REPO / entry["folder"] / "solution.py"
        loc = 0
        if sol.exists():
            loc = sum(1 for _ in open(sol))
        total += loc
        per_day.append((entry["day"], entry["folder"], loc))
    return total, per_day


# ── SVG heatmap ──────────────────────────────────────────────────────────────

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def generate_heatmap_svg(progress):
    """Create a GitHub-contribution-calendar-style SVG and save to docs/heatmap.svg."""
    # Build date -> track lookup
    date_track = {}
    for entry in progress["days"]:
        d = datetime.strptime(entry["date"], "%Y-%m-%d").date()
        # If multiple entries on same date, last one wins (or could merge; keep simple)
        date_track[d] = entry["track"]

    # Determine grid range: START_DATE to today (or last entry, whichever is later)
    today = date.today()
    all_dates = list(date_track.keys()) + [today]
    end_date = max(all_dates)

    # Align start to Monday of the week containing START_DATE
    start_monday = START_DATE - timedelta(days=START_DATE.weekday())  # Monday

    # Number of weeks (columns)
    total_days = (end_date - start_monday).days + 1
    num_weeks = (total_days + 6) // 7  # ceil
    # Ensure at least a few weeks for visual appeal
    num_weeks = max(num_weeks, 13)

    cell = 14
    gap = 3
    step = cell + gap
    margin_left = 32
    margin_top = 28
    legend_height = 40

    svg_w = margin_left + num_weeks * step + 20
    svg_h = margin_top + 7 * step + legend_height + 10

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}"'
        f' viewBox="0 0 {svg_w} {svg_h}">'
    )
    parts.append(
        f'<rect width="{svg_w}" height="{svg_h}" fill="{COLOR_BG}" rx="6"/>'
    )

    # Day-of-week labels (Mon, Wed, Fri)
    day_labels = {0: "Mon", 2: "Wed", 4: "Fri"}
    for row, label in day_labels.items():
        y = margin_top + row * step + cell - 2
        parts.append(
            f'<text x="2" y="{y}" fill="{COLOR_TEXT}" '
            f'font-family="monospace" font-size="10">{label}</text>'
        )

    # Month labels
    prev_month = -1
    for week in range(num_weeks):
        d = start_monday + timedelta(weeks=week)
        if d.month != prev_month:
            prev_month = d.month
            x = margin_left + week * step
            parts.append(
                f'<text x="{x}" y="{margin_top - 8}" fill="{COLOR_TEXT}" '
                f'font-family="monospace" font-size="10">{MONTH_NAMES[d.month - 1]}</text>'
            )

    # Grid squares
    for week in range(num_weeks):
        for dow in range(7):
            d = start_monday + timedelta(weeks=week, days=dow)
            if d < START_DATE or d > end_date:
                color = COLOR_BG  # invisible — outside range
            elif d in date_track:
                color = TRACK_COLORS.get(date_track[d], COLOR_EMPTY)
            else:
                color = COLOR_EMPTY

            if d < START_DATE or d > end_date:
                continue  # skip drawing squares outside range

            x = margin_left + week * step
            y = margin_top + dow * step
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'rx="3" fill="{color}"/>'
            )

    # Legend
    legend_y = margin_top + 7 * step + 14
    legend_x = margin_left
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" fill="{COLOR_TEXT}" '
        f'font-family="monospace" font-size="10">Tracks:</text>'
    )
    lx = legend_x + 52
    for track, color in TRACK_COLORS.items():
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 10}" width="{cell}" height="{cell}" '
            f'rx="3" fill="{color}"/>'
        )
        lx += cell + 4
        parts.append(
            f'<text x="{lx}" y="{legend_y}" fill="{COLOR_TEXT_BRIGHT}" '
            f'font-family="monospace" font-size="10">{track.title()}</text>'
        )
        lx += len(track) * 7 + 12
    # Empty square
    parts.append(
        f'<rect x="{lx}" y="{legend_y - 10}" width="{cell}" height="{cell}" '
        f'rx="3" fill="{COLOR_EMPTY}"/>'
    )
    lx += cell + 4
    parts.append(
        f'<text x="{lx}" y="{legend_y}" fill="{COLOR_TEXT_BRIGHT}" '
        f'font-family="monospace" font-size="10">No activity</text>'
    )

    parts.append("</svg>")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_PATH.write_text("\n".join(parts) + "\n")


# ── Progress bars ────────────────────────────────────────────────────────────


def progress_bar(done, total, width=20):
    """Return a Unicode progress bar string like ████████░░░░░░░░ 8/24."""
    filled = round(done / total * width) if total > 0 else 0
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"{bar} {done}/{total}"


# ── Badge markdown ───────────────────────────────────────────────────────────


def make_badges(total_days, current_streak, longest_streak):
    today_str = date.today().strftime("%Y--%m--%d")
    badges = [
        f"![Days](https://img.shields.io/badge/days-{total_days}-blue)",
        f"![Streak](https://img.shields.io/badge/streak-{current_streak}_days-orange)",
        f"![Longest](https://img.shields.io/badge/longest-{longest_streak}_days-green)",
        f"![Updated](https://img.shields.io/badge/last_updated-{today_str}-lightgrey)",
    ]
    return " ".join(badges)


# ── README section replacement ───────────────────────────────────────────────


def replace_section(text, name, content):
    """Replace content between <!-- NAME:START --> and <!-- NAME:END --> markers.

    If markers don't exist, insert them at a sensible location.
    """
    start_marker = f"<!-- {name}:START -->"
    end_marker = f"<!-- {name}:END -->"

    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )

    block = f"{start_marker}\n{content}\n{end_marker}"

    if pattern.search(text):
        return pattern.sub(block, text)
    else:
        # Markers don't exist yet — insert at the right place
        return insert_section(text, name, block)


def insert_section(text, name, block):
    """Insert a new marker block at a logical position in the README."""
    # Insertion order preference:
    # BADGES -> right after the H1 line
    # HEATMAP -> before "## Dashboard" or after Progression section
    # STREAK -> after HEATMAP
    # PROGRESS_BARS -> after STREAK

    if name == "BADGES":
        # After the first heading line
        m = re.search(r"^# .+\n", text)
        if m:
            pos = m.end()
            return text[:pos] + "\n" + block + "\n" + text[pos:]

    if name == "HEATMAP":
        # Before "## Dashboard"
        m = re.search(r"^## Dashboard", text, re.MULTILINE)
        if m:
            return text[: m.start()] + block + "\n\n" + text[m.start():]

    if name == "STREAK":
        # After HEATMAP end marker
        m = re.search(r"<!-- HEATMAP:END -->", text)
        if m:
            pos = m.end()
            return text[:pos] + "\n\n" + block + "\n" + text[pos:]

    if name == "PROGRESS_BARS":
        # After STREAK end marker
        m = re.search(r"<!-- STREAK:END -->", text)
        if m:
            pos = m.end()
            return text[:pos] + "\n\n" + block + "\n" + text[pos:]

    # Fallback: append before ## Dashboard or at end
    m = re.search(r"^## Dashboard", text, re.MULTILINE)
    if m:
        return text[: m.start()] + block + "\n\n" + text[m.start():]
    return text + "\n\n" + block + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    progress = load_progress()
    curriculum = load_curriculum()

    # Compute stats
    total_days = len(progress["days"])
    current_streak, longest_streak = compute_streaks(progress)
    track_prog = compute_track_progress(progress, curriculum)
    total_loc, per_day_loc = compute_loc(progress)

    # Generate heatmap SVG
    generate_heatmap_svg(progress)

    # Build section contents
    badges_content = make_badges(total_days, current_streak, longest_streak)

    heatmap_content = "![Contribution Heatmap](docs/heatmap.svg)"

    streak_content = (
        f"### \U0001f525 Current Streak: {current_streak} days | "
        f"Longest: {longest_streak} days"
    )

    # Progress bars
    bar_lines = []
    display_order = ["ai", "crypto", "robotics", "integration"]
    track_labels = {
        "ai": "AI         ",
        "crypto": "Crypto     ",
        "robotics": "Robotics   ",
        "integration": "Integration",
    }
    for t in display_order:
        done, total = track_prog.get(t, (0, 0))
        label = track_labels[t]
        bar = progress_bar(done, total)
        bar_lines.append(f"**{label}** `{bar}`  ")

    # LOC summary
    bar_lines.append("")
    bar_lines.append(f"**Total LOC**: {total_loc:,} lines across {total_days} solutions")

    progress_bars_content = "\n".join(bar_lines)

    # Read and update README
    readme_text = README_PATH.read_text()

    readme_text = replace_section(readme_text, "BADGES", badges_content)
    readme_text = replace_section(readme_text, "HEATMAP", heatmap_content)
    readme_text = replace_section(readme_text, "STREAK", streak_content)
    readme_text = replace_section(readme_text, "PROGRESS_BARS", progress_bars_content)

    README_PATH.write_text(readme_text)
    print(f"README.md updated  ({total_days} days, streak {current_streak})")
    print(f"docs/heatmap.svg generated  ({total_loc:,} LOC total)")


if __name__ == "__main__":
    main()
