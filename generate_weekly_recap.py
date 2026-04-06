#!/usr/bin/env python3
"""generate_weekly_recap.py — produces a weekly recap markdown file for build-everyday.

Run on Sundays (or any day with --force). Creates recaps/WEEK-XX.md summarizing the
week's completed challenges, key concepts, cross-track connections, and stats.

Usage:
    python3 generate_weekly_recap.py            # only runs on Sunday
    python3 generate_weekly_recap.py --force     # run any day, regenerate even if exists
    python3 generate_weekly_recap.py --dry-run   # print to stdout, don't write file
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(REPO_DIR, "progress.json")
RECAPS_DIR = os.path.join(REPO_DIR, "recaps")
START_DATE = datetime(2026, 4, 1).date()

TRACK_LABEL = {
    "ai": "AI",
    "crypto": "Crypto",
    "robotics": "Robotics",
    "integration": "Integration",
}


def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def get_week_number(date):
    """Return the 1-based week number since the start of the challenge."""
    delta = (date - START_DATE).days
    return (delta // 7) + 1


def get_week_bounds(week_num):
    """Return (monday, sunday) date range for the given week number."""
    # Week 1 starts on the Monday of the week containing START_DATE.
    # START_DATE is 2026-04-01 (Wednesday). Week 1 Monday = 2026-03-30.
    # But since the challenge starts Apr 1, we define weeks relative to START_DATE.
    # Week 1 = days 0-6 from START_DATE, etc.
    week_start = START_DATE + timedelta(days=(week_num - 1) * 7)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def get_current_week_number():
    """Return the week number for today."""
    today = datetime.now().date()
    return get_week_number(today)


def find_week_challenges(days, week_num):
    """Return all challenges completed during the given week."""
    week_start, week_end = get_week_bounds(week_num)
    result = []
    for d in days:
        date = datetime.strptime(d["date"], "%Y-%m-%d").date()
        if week_start <= date <= week_end:
            result.append(d)
    return result


def read_readme_section(folder, section_names):
    """Read specific sections from a day's README.md. Returns the section text or empty string."""
    readme_path = os.path.join(REPO_DIR, folder, "README.md")
    if not os.path.exists(readme_path):
        return ""

    with open(readme_path) as f:
        content = f.read()

    # Find sections by heading
    for name in section_names:
        # Match ## Section Name or ### Section Name (case-insensitive)
        pattern = rf'(?:^|\n)##+ *{re.escape(name)}[^\n]*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


def extract_key_concepts(folder):
    """Extract key concepts from a day's README.md."""
    # Try several common section names
    text = read_readme_section(folder, [
        "Core Concepts",
        "Core concepts",
        "Key Concepts",
        "Key concepts",
    ])
    if not text:
        text = read_readme_section(folder, [
            "Learning Objectives",
            "Learning objectives",
        ])

    if not text:
        return []

    # Extract subsection headings as concept names
    concepts = []
    for match in re.finditer(r'^###+ *(.+)', text, re.MULTILINE):
        concept = match.group(1).strip()
        # Clean up any markdown formatting
        concept = re.sub(r'[*_`]', '', concept)
        concepts.append(concept)

    # If no subsections, extract bullet points or first few sentences
    if not concepts:
        for match in re.finditer(r'^[-*] +(.+)', text, re.MULTILINE):
            line = match.group(1).strip()
            line = re.sub(r'[*_`]', '', line)
            if len(line) > 10:
                concepts.append(line)
            if len(concepts) >= 5:
                break

    return concepts


def extract_learning_objectives(folder):
    """Extract learning objectives from a day's README.md."""
    text = read_readme_section(folder, [
        "Learning Objectives",
        "Learning objectives",
    ])
    if not text:
        return []

    objectives = []
    for match in re.finditer(r'^[-*] +(.+)', text, re.MULTILINE):
        obj = match.group(1).strip()
        obj = re.sub(r'[*_`]', '', obj)
        objectives.append(obj)

    return objectives


def count_loc(folder):
    """Count lines of Python code in a day's folder."""
    folder_path = os.path.join(REPO_DIR, folder)
    if not os.path.isdir(folder_path):
        return 0

    total = 0
    for fname in os.listdir(folder_path):
        if fname.endswith(".py"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath) as f:
                total += sum(1 for line in f if line.strip())
    return total


def find_cross_track_connections(challenges_by_track):
    """Look for overlapping concepts between tracks."""
    connections = []

    # Collect all concepts per track
    track_concepts = {}
    for track, challenges in challenges_by_track.items():
        concepts_set = set()
        for c in challenges:
            for concept in c.get("_concepts", []):
                # Normalize for comparison
                normalized = concept.lower()
                concepts_set.add(normalized)
            for obj in c.get("_objectives", []):
                normalized = obj.lower()
                concepts_set.add(normalized)
        track_concepts[track] = concepts_set

    # Common keywords that indicate connections
    connection_keywords = [
        ("optimization", "gradient", "loss function", "minimize"),
        ("hash", "cryptograph", "digest", "sha"),
        ("tree", "graph", "node"),
        ("probability", "bayes", "distribution", "likelihood"),
        ("matrix", "vector", "linear algebra"),
        ("neural network", "backpropagation", "forward pass"),
        ("cluster", "classification", "segmentation"),
        ("reinforcement", "reward", "policy", "agent"),
    ]

    tracks = list(track_concepts.keys())
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            t1, t2 = tracks[i], tracks[j]
            all_t1 = " ".join(track_concepts[t1])
            all_t2 = " ".join(track_concepts[t2])

            for keyword_group in connection_keywords:
                t1_match = any(kw in all_t1 for kw in keyword_group)
                t2_match = any(kw in all_t2 for kw in keyword_group)
                if t1_match and t2_match:
                    # Find the specific matching keywords
                    matched = [kw for kw in keyword_group if kw in all_t1 and kw in all_t2]
                    if matched:
                        connections.append(
                            f"{TRACK_LABEL[t1]} and {TRACK_LABEL[t2]} both explore "
                            f"concepts related to: {', '.join(matched)}"
                        )

    return connections


def generate_recap(week_num, challenges, force=False):
    """Generate the markdown recap content."""
    week_start, week_end = get_week_bounds(week_num)
    date_range = f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"

    # Group by track
    by_track = defaultdict(list)
    for c in challenges:
        by_track[c["track"]].append(c)

    # Enrich with concepts and objectives
    total_loc = 0
    for c in challenges:
        c["_concepts"] = extract_key_concepts(c["folder"])
        c["_objectives"] = extract_learning_objectives(c["folder"])
        c["_loc"] = count_loc(c["folder"])
        total_loc += c["_loc"]

    # Find cross-track connections
    connections = find_cross_track_connections(by_track)

    # ── Build Markdown ───────────────────────────────────────────────────

    lines = []
    lines.append(f"# Week {week_num} Recap")
    lines.append("")
    lines.append(f"**{date_range}** | {len(challenges)} challenges completed | {total_loc} lines of code")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary by track
    lines.append("## Challenges by Track")
    lines.append("")

    for track in ["ai", "crypto", "robotics", "integration"]:
        if track not in by_track:
            continue
        label = TRACK_LABEL[track]
        lines.append(f"### {label}")
        lines.append("")
        for c in by_track[track]:
            day_str = f"Day {c['day']:03d}"
            date = datetime.strptime(c["date"], "%Y-%m-%d").strftime("%a %b %d")
            lines.append(f"- **{day_str}: {c['challenge']}** ({date})")
            if c["_loc"]:
                lines.append(f"  - {c['_loc']} lines of code")
        lines.append("")

    # Key concepts
    lines.append("## Key Concepts Covered")
    lines.append("")

    for track in ["ai", "crypto", "robotics", "integration"]:
        if track not in by_track:
            continue
        label = TRACK_LABEL[track]
        all_concepts = []
        for c in by_track[track]:
            all_concepts.extend(c["_concepts"])
        if all_concepts:
            lines.append(f"### {label}")
            lines.append("")
            for concept in all_concepts:
                lines.append(f"- {concept}")
            lines.append("")

    # Learning objectives
    has_objectives = any(c["_objectives"] for c in challenges)
    if has_objectives:
        lines.append("## Learning Objectives")
        lines.append("")
        for c in challenges:
            if c["_objectives"]:
                lines.append(f"**Day {c['day']:03d} — {c['challenge']}:**")
                for obj in c["_objectives"]:
                    lines.append(f"- {obj}")
                lines.append("")

    # Cross-track connections
    if connections:
        lines.append("## Cross-Track Connections")
        lines.append("")
        for conn in connections:
            lines.append(f"- {conn}")
        lines.append("")

    # Stats
    lines.append("## Stats")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Challenges completed | {len(challenges)} |")
    lines.append(f"| Total lines of code | {total_loc} |")
    for track in ["ai", "crypto", "robotics", "integration"]:
        if track in by_track:
            lines.append(f"| {TRACK_LABEL[track]} challenges | {len(by_track[track])} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate weekly recap for build-everyday")
    parser.add_argument("--force", action="store_true", help="Run even if not Sunday; regenerate existing recaps")
    parser.add_argument("--dry-run", action="store_true", help="Print recap to stdout instead of writing file")
    parser.add_argument("--week", type=int, default=None, help="Generate recap for a specific week number")
    args = parser.parse_args()

    today = datetime.now().date()
    is_sunday = today.weekday() == 6  # 6 = Sunday

    if not is_sunday and not args.force:
        print("Today is not Sunday. Use --force to generate a recap anyway.")
        sys.exit(0)

    # Determine which week to recap
    if args.week is not None:
        week_num = args.week
    else:
        week_num = get_current_week_number()

    # Output file path
    recap_filename = f"WEEK-{week_num:02d}.md"
    recap_path = os.path.join(RECAPS_DIR, recap_filename)

    # Idempotency check
    if os.path.exists(recap_path) and not args.force:
        print(f"Recap already exists: {recap_path}")
        print("Use --force to regenerate.")
        sys.exit(0)

    # Load progress and find this week's challenges
    progress = load_progress()
    challenges = find_week_challenges(progress["days"], week_num)

    if not challenges:
        print(f"No challenges found for week {week_num}.")
        sys.exit(0)

    # Generate the recap
    content = generate_recap(week_num, challenges, force=args.force)

    if args.dry_run:
        print(content)
    else:
        os.makedirs(RECAPS_DIR, exist_ok=True)
        with open(recap_path, "w") as f:
            f.write(content)
        print(f"Weekly recap written to: {recap_path}")


if __name__ == "__main__":
    main()
