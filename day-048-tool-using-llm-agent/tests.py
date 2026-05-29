"""
Day 48: Tool-Using LLM Agent — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import json
import unittest
from my_solution import (
    ActionType,
    Agent,
    AgentStep,
    CalculatorTool,
    KnowledgeBaseTool,
    LLMSimulator,
    ParsedAction,
    ParameterSpec,
    StringTool,
    ToolRegistry,
    ToolResult,
    WeatherTool,
    parse_llm_output,
)


class TestParameterValidation(unittest.TestCase):
    """Test the Tool.validate_params method."""

    def test_valid_params(self):
        tool = CalculatorTool()
        result = tool.validate_params({"expression": "1 + 1"})
        self.assertIsNone(result)

    def test_missing_required_param(self):
        tool = CalculatorTool()
        result = tool.validate_params({})
        self.assertIsNotNone(result)
        self.assertIn("expression", result)

    def test_unknown_param(self):
        tool = CalculatorTool()
        result = tool.validate_params({"expression": "1+1", "bogus": "value"})
        self.assertIsNotNone(result)
        self.assertIn("bogus", result)

    def test_enum_validation(self):
        tool = StringTool()
        result = tool.validate_params({"text": "hello", "operation": "invalid_op"})
        self.assertIsNotNone(result)

    def test_valid_enum(self):
        tool = StringTool()
        result = tool.validate_params({"text": "hello", "operation": "reverse"})
        self.assertIsNone(result)


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        result = self.tool.execute(expression="2 + 3")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "5")

    def test_complex_expression(self):
        result = self.tool.execute(expression="(17 * 3) + sqrt(144)")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "63.0")

    def test_dangerous_input_blocked(self):
        result = self.tool.execute(expression="__import__('os').system('ls')")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


class TestWeatherTool(unittest.TestCase):
    def setUp(self):
        self.tool = WeatherTool()

    def test_known_city(self):
        result = self.tool.execute(city="Tokyo")
        self.assertTrue(result.success)
        self.assertIn("68", result.output)

    def test_unknown_city(self):
        result = self.tool.execute(city="Atlantis")
        self.assertFalse(result.success)


class TestKnowledgeBaseTool(unittest.TestCase):
    def setUp(self):
        self.tool = KnowledgeBaseTool()

    def test_matching_query(self):
        result = self.tool.execute(query="python creator")
        self.assertTrue(result.success)
        self.assertIn("Guido", result.output)

    def test_no_match(self):
        result = self.tool.execute(query="xyzzy nonsense")
        self.assertFalse(result.success)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())
        self.registry.register(WeatherTool())

    def test_register_and_get(self):
        tool = self.registry.get("calculator")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "calculator")

    def test_get_unknown(self):
        tool = self.registry.get("nonexistent")
        self.assertIsNone(tool)

    def test_duplicate_registration_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register(CalculatorTool())

    def test_tool_names(self):
        names = self.registry.tool_names
        self.assertIn("calculator", names)
        self.assertIn("weather", names)

    def test_execute_tool(self):
        result = self.registry.execute_tool("calculator", {"expression": "5 * 5"})
        self.assertTrue(result.success)
        self.assertEqual(result.output, "25")

    def test_execute_unknown_tool(self):
        result = self.registry.execute_tool("nonexistent", {})
        self.assertFalse(result.success)

    def test_generate_prompt_contains_tools(self):
        prompt = self.registry.generate_tool_prompt()
        self.assertIn("calculator", prompt)
        self.assertIn("weather", prompt)


class TestActionParser(unittest.TestCase):
    def test_parse_tool_call(self):
        output = (
            "Thought: I need to calculate something.\n"
            "Action: calculator\n"
            'Action Input: {"expression": "2 + 2"}'
        )
        parsed = parse_llm_output(output)
        self.assertEqual(parsed.action_type, ActionType.TOOL_CALL)
        self.assertEqual(parsed.tool_name, "calculator")
        self.assertEqual(parsed.tool_params["expression"], "2 + 2")
        self.assertIn("calculate", parsed.thought.lower())

    def test_parse_final_answer(self):
        output = "Thought: I know the answer.\nFinal Answer: The result is 42."
        parsed = parse_llm_output(output)
        self.assertEqual(parsed.action_type, ActionType.FINAL_ANSWER)
        self.assertIn("42", parsed.final_answer)

    def test_parse_error_no_action(self):
        output = "I don't know what to do."
        parsed = parse_llm_output(output)
        self.assertEqual(parsed.action_type, ActionType.PARSE_ERROR)

    def test_parse_error_bad_json(self):
        output = "Thought: test\nAction: calc\nAction Input: not json"
        parsed = parse_llm_output(output)
        self.assertEqual(parsed.action_type, ActionType.PARSE_ERROR)


class TestAgent(unittest.TestCase):
    """Integration tests for the full agent loop."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())
        self.registry.register(WeatherTool())
        self.registry.register(KnowledgeBaseTool())
        self.registry.register(StringTool())
        self.llm = LLMSimulator()

    def test_calculator_query(self):
        agent = Agent(self.registry, self.llm, max_steps=5, verbose=False)
        answer, steps = agent.run("Calculate 2 + 2")
        self.assertIn("4", answer)
        self.assertTrue(len(steps) >= 2)  # At least: tool call + final answer

    def test_multi_step_weather_query(self):
        agent = Agent(self.registry, self.llm, max_steps=5, verbose=False)
        answer, steps = agent.run("What's the weather like at Apple's headquarters?")
        self.assertIn("Cupertino", answer)
        self.assertTrue(len(steps) >= 3)  # knowledge search + weather + final answer

    def test_max_steps_respected(self):
        agent = Agent(self.registry, self.llm, max_steps=1, verbose=False)
        answer, steps = agent.run("Calculate 1 + 1")
        self.assertTrue(len(steps) <= 1)


if __name__ == "__main__":
    unittest.main()
