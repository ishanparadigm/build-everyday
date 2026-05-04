"""
Day 033: Prompt Chaining Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py

Tests import from my_solution — implement your solution there.
"""

import asyncio
import json
import unittest
from my_solution import (
    ChainStep,
    ChainRunner,
    ChainResult,
    StepResult,
    StepStatus,
    SimulatedLLM,
    build_research_chain,
    validate_json_with_keys,
    validate_min_length,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestValidators(unittest.TestCase):
    """Test the validation functions that guard chain step outputs."""

    def test_json_validator_accepts_valid_json(self):
        validator = validate_json_with_keys(["name", "age"])
        is_valid, error = validator('{"name": "Alice", "age": 30}')
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_json_validator_rejects_invalid_json(self):
        validator = validate_json_with_keys(["name"])
        is_valid, error = validator("not json at all")
        self.assertFalse(is_valid)
        self.assertIn("JSON", error)

    def test_json_validator_rejects_missing_keys(self):
        validator = validate_json_with_keys(["name", "age", "email"])
        is_valid, error = validator('{"name": "Alice"}')
        self.assertFalse(is_valid)
        self.assertIn("age", error)
        self.assertIn("email", error)

    def test_min_length_accepts_long_text(self):
        validator = validate_min_length(10)
        is_valid, error = validator("This is definitely long enough text")
        self.assertTrue(is_valid)

    def test_min_length_rejects_short_text(self):
        validator = validate_min_length(100)
        is_valid, error = validator("Too short")
        self.assertFalse(is_valid)
        self.assertIn("too short", error.lower())


class TestSimulatedLLM(unittest.TestCase):
    """Test the simulated LLM backend."""

    def test_returns_response_and_token_count(self):
        llm = SimulatedLLM(latency_ms=1)
        text, tokens = run_async(
            llm.call("system", "user prompt", "research_planner")
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    def test_planner_returns_valid_json_with_questions(self):
        llm = SimulatedLLM(latency_ms=1)
        text, _ = run_async(
            llm.call("system", "some topic", "research_planner")
        )
        data = json.loads(text)
        self.assertIn("questions", data)
        self.assertIsInstance(data["questions"], list)
        self.assertGreater(len(data["questions"]), 0)

    def test_researcher_returns_valid_json_with_findings(self):
        llm = SimulatedLLM(latency_ms=1)
        text, _ = run_async(
            llm.call("system", "some question", "researcher")
        )
        data = json.loads(text)
        self.assertIn("findings", data)
        self.assertIn("confidence", data)


class TestChainStep(unittest.TestCase):
    """Test individual chain step execution."""

    def test_successful_step(self):
        llm = SimulatedLLM(latency_ms=1)
        step = ChainStep(
            name="research_planner",
            system_prompt="Plan research",
            input_formatter=lambda x: f"Research: {x}",
            output_validator=validate_json_with_keys(["questions"]),
        )
        result = run_async(step.execute(llm, "AI systems"))
        self.assertEqual(result.status, StepStatus.SUCCESS)
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.tokens_used, 0)
        self.assertEqual(result.retries, 0)

    def test_step_tracks_metrics(self):
        llm = SimulatedLLM(latency_ms=1)
        step = ChainStep(
            name="synthesizer",
            system_prompt="Synthesize",
            input_formatter=lambda x: x,
            output_validator=validate_min_length(10),
        )
        result = run_async(step.execute(llm, "findings data"))
        self.assertIsInstance(result.latency_ms, float)
        self.assertIsInstance(result.tokens_used, int)
        self.assertIsInstance(result.step_name, str)


class TestChainRunner(unittest.TestCase):
    """Test the full chain execution."""

    def test_full_chain_produces_result(self):
        steps = build_research_chain()
        runner = ChainRunner(llm=SimulatedLLM(latency_ms=1))
        result = run_async(runner.run("test topic", steps))
        self.assertIsInstance(result, ChainResult)
        self.assertGreater(len(result.final_output), 0)
        self.assertGreater(len(result.steps), 0)

    def test_chain_tracks_total_metrics(self):
        steps = build_research_chain()
        runner = ChainRunner(llm=SimulatedLLM(latency_ms=1))
        result = run_async(runner.run("test topic", steps))
        self.assertGreater(result.total_latency_ms, 0)
        self.assertGreater(result.total_tokens, 0)

    def test_chain_has_correct_number_of_steps(self):
        """Chain should have 5+ steps: plan, N research, synth, critic, edit."""
        steps = build_research_chain()
        runner = ChainRunner(llm=SimulatedLLM(latency_ms=1))
        result = run_async(runner.run("test topic", steps))
        # At minimum: 1 plan + 1 research + 1 synth + 1 critic + 1 edit = 5
        # With 3 parallel research calls: 1 + 3 + 1 + 1 + 1 = 7
        self.assertGreaterEqual(len(result.steps), 5)

    def test_chain_summary_is_readable(self):
        steps = build_research_chain()
        runner = ChainRunner(llm=SimulatedLLM(latency_ms=1))
        result = run_async(runner.run("test topic", steps))
        summary = result.summary()
        self.assertIn("Step", summary)
        self.assertIn("tokens", summary.lower())


class TestBuildResearchChain(unittest.TestCase):
    """Test the chain construction function."""

    def test_returns_five_steps(self):
        steps = build_research_chain()
        self.assertEqual(len(steps), 5)

    def test_steps_have_names(self):
        steps = build_research_chain()
        names = [s.name for s in steps]
        self.assertIn("research_planner", names)
        self.assertIn("researcher", names)
        self.assertIn("synthesizer", names)
        self.assertIn("critic", names)
        self.assertIn("final_editor", names)


if __name__ == "__main__":
    unittest.main()
