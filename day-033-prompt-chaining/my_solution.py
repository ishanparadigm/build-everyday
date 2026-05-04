"""
Day 033: Prompt Chaining with Claude API — Your Implementation

Build a multi-step prompt chaining pipeline that decomposes a complex task
into sequential LLM calls, where each step's output feeds the next step's input.

Key concepts to implement:
- ChainStep: encapsulates a single step (system prompt, formatter, validator, retries)
- Validators: ensure step outputs meet expectations before passing downstream
- ChainRunner: orchestrates sequential + parallel execution with logging
- SimulatedLLM: mock backend so you can test without API keys

Hints:
- Each step needs a clear input/output contract (think: function signatures for LLMs)
- Validation between steps catches errors early — parse JSON, check required keys
- Use asyncio.gather() for parallel execution of independent sub-tasks
- Track metrics (latency, tokens, retries) per step for observability
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    """Possible outcomes for a chain step."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRIED = "retried"


@dataclass
class StepResult:
    """Captures everything about a single step's execution."""
    step_name: str
    input_text: str
    output_text: str
    status: StepStatus
    latency_ms: float
    tokens_used: int
    retries: int = 0
    error: Optional[str] = None


@dataclass
class ChainResult:
    """The full result of running a chain, including all intermediate steps."""
    final_output: str
    steps: list[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    def summary(self) -> str:
        """Human-readable summary of the chain execution."""
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Simulated LLM
# Hint: Build a class that returns pre-built responses keyed by step_name.
# This lets you test the chain logic without an API key. The interface
# should match what you'd use with the real Anthropic SDK:
#   async def call(system_prompt, user_prompt, step_name) -> (text, tokens)
# ---------------------------------------------------------------------------

class SimulatedLLM:
    """Simulates Claude API responses for testing.

    Hint: Store response generators in a dict keyed by step_name.
    Each generator takes the user_prompt and returns a response string.
    Simulate latency with asyncio.sleep().
    """

    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
        raise NotImplementedError("TODO: implement this — register response generators")

    async def call(self, system_prompt: str, user_prompt: str,
                   step_name: str) -> tuple[str, int]:
        """Simulate an LLM API call. Returns (response_text, token_count)."""
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Chain Step
# Hint: The key method is execute(). It should:
# 1. Format the input using input_formatter
# 2. Call the LLM
# 3. Validate the output
# 4. If invalid, retry with the validation error appended to the prompt
# 5. Return a StepResult with metrics
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """A single step in a prompt chain."""
    name: str
    system_prompt: str
    input_formatter: Callable[[str], str]
    output_validator: Callable[[str], tuple[bool, str]] = field(
        default_factory=lambda: lambda x: (True, "")
    )
    max_retries: int = 2

    async def execute(self, llm: SimulatedLLM, input_text: str) -> StepResult:
        """Execute this step with retry logic.

        Hint: On retry, append the validation error to the prompt so the
        LLM knows what to fix. This is much more effective than blind retries.
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Validators
# Hint: These are factory functions that return validator callables.
# A validator takes a string and returns (is_valid: bool, error_msg: str).
# ---------------------------------------------------------------------------

def validate_json_with_keys(required_keys: list[str]) -> Callable[[str], tuple[bool, str]]:
    """Create a validator that checks for valid JSON with specific keys."""
    raise NotImplementedError("TODO: implement this")


def validate_min_length(min_chars: int) -> Callable[[str], tuple[bool, str]]:
    """Ensure output meets a minimum length threshold."""
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Build the Research Chain
# Hint: Create 5 ChainStep objects:
#   1. research_planner — topic → JSON with "questions" list
#   2. researcher — question → JSON with "question", "findings", "confidence"
#   3. synthesizer — all findings → markdown analysis
#   4. critic — analysis → JSON with "strengths", "gaps", "suggestions", "overall_rating"
#   5. final_editor — analysis + critique → polished markdown
# ---------------------------------------------------------------------------

def build_research_chain() -> list[ChainStep]:
    """Build the 5-step research and analysis chain."""
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Chain Runner
# Hint: The run() method orchestrates the pipeline:
# 1. Run planner → get questions
# 2. Run researcher on EACH question IN PARALLEL (asyncio.gather)
# 3. Collect findings → run synthesizer
# 4. Run critic on the synthesis
# 5. Combine synthesis + critique → run final_editor
# Track StepResults for every call and build ChainResult at the end.
# ---------------------------------------------------------------------------

class ChainRunner:
    """Orchestrates execution of a prompt chain."""

    def __init__(self, llm: Optional[SimulatedLLM] = None):
        self.llm = llm or SimulatedLLM()
        self.results: list[StepResult] = []

    async def run(self, topic: str, steps: list[ChainStep]) -> ChainResult:
        """Execute the research chain on a given topic."""
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Main — test your implementation
# ---------------------------------------------------------------------------

async def main():
    """Run the research chain and display results."""
    # Build the chain
    steps = build_research_chain()
    print(f"Built chain with {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step.name}")

    # Create the runner
    runner = ChainRunner(llm=SimulatedLLM(latency_ms=30))

    # Execute
    topic = "prompt chaining in production AI systems"
    result = await runner.run(topic, steps)

    # Display
    print("\n" + result.summary())
    print("\nFINAL OUTPUT:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
