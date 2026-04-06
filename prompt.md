You are maintaining a daily coding challenge repo designed to build deep technical mastery in AI, crypto, and robotics.

Steps:
1. Read curriculum.json to get the track schedule and challenge lists.
2. Read progress.json to see which challenges have already been completed. Do NOT pick a challenge that already appears in progress.json.
3. Determine today's track: Mon/Thu = AI, Tue/Fri = Crypto, Wed/Sat = Robotics, Sun = Integration.
4. Count existing day-XXX-* folders to determine the next day number (zero-padded to 3 digits).
5. Determine the current week number (day_number / 7 + 1) to pick the difficulty tier (weeks 1-4, 5-8, 9-12, or 13+).
6. Pick the next unused challenge from the appropriate track and week tier in curriculum.json. If all challenges in a tier are used, create a new related challenge at the same difficulty level.
7. Create a folder day-XXX-short-topic-name/ containing:
   - README.md with:
     - Challenge title and overview — what you're building and why it matters in the real world
     - Core concepts — the key theoretical ideas behind this challenge, explained from first principles. Don't just name them — teach them. Include the math, the intuition, and the tradeoffs.
     - Step-by-step breakdown — walk through the approach in detail before any code. Explain each step: what it does, why it's necessary, and what would go wrong without it.
     - Learning objectives — specific technical skills gained
     - Going deeper — pointers to advanced variations, edge cases, and how this concept connects to production systems
   - solution.py with:
     - A complete, production-quality Python implementation
     - Extensive inline comments explaining each step — not just WHAT the code does, but WHY. Explain algorithmic choices, data structure decisions, and complexity tradeoffs.
     - Type hints and clear function signatures
     - A if __name__ == '__main__' block with example usage that demonstrates the solution working, prints intermediate steps so the reader can follow the execution, and shows expected output
     - Where relevant: performance analysis, edge case handling, and comparison to alternative approaches in comments
   - my_solution.py with:
     - A skeleton template for the user to implement themselves
     - All function/class signatures with type hints and docstrings copied from solution.py
     - `raise NotImplementedError("TODO: implement this")` as the body for each function
     - Hints as comments pointing to key concepts (but NOT the implementation)
     - A if __name__ == '__main__' block that exercises the functions so the user can test as they build
   - tests.py with:
     - A comprehensive test suite using unittest that imports from my_solution (NOT solution)
     - 5-10 tests covering core functionality, edge cases, and correctness
     - A docstring at the top explaining how to run: `python3 -m pytest tests.py` or `python3 tests.py`
     - Tests should verify the algorithm works correctly, not just that it runs
8. Run the solution: execute `python3 solution.py` in the new folder and verify it exits with code 0. If it fails, fix the code and re-run until it passes.
9. Update progress.json: add an entry for this day with day_number, track, challenge_name, folder_name, and date.
10. Update the Progress section at the bottom of the root README.md — regenerate the dashboard table AND append the new day entry to the bullet list.
11. Git add all new/changed files, commit with message 'Day XXX: [Topic]', and push to origin main.

The goal is TECHNICAL MASTERY — not just completing exercises. Each day's work should leave the reader with a deep understanding of the underlying concepts, not just a working script. Write as if teaching a smart engineer who wants to truly understand the material, not just copy-paste a solution. Build on concepts from previous days where possible to show progression.
