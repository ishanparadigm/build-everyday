"""
Day 48: Tool-Using LLM Agent — Your Implementation

Build the ReAct agent framework from scratch: tool registry, action parsing,
execution engine, and the core agent loop.

Hints:
- Start with the Tool ABC and ParameterSpec — get the interface right first
- Build one concrete tool (Calculator) and test it standalone before building the registry
- The action parser is the trickiest part — test it with hardcoded strings
- The agent loop is simple once you have parsing + execution working
- The LLM simulator is last — it's just if/else rules to exercise the framework
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Part 1: Tool Interface and Registry
# =============================================================================

@dataclass
class ParameterSpec:
    """Schema for a single tool parameter.

    Fields: name, type ("string"/"number"/"boolean"), description, required (bool),
    enum (optional list of allowed values).
    """
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolResult:
    """Result of executing a tool.

    Fields: success (bool), output (str), error (str | None).
    """
    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    """Abstract base class for all tools.

    Hint: Tools are self-describing — they carry their own name, description,
    and parameter schema. Think of this as defining a plugin interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description the LLM reads to decide when to use this tool."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ParameterSpec]:
        """Parameter schema for this tool."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with validated parameters."""
        ...

    def validate_params(self, params: dict[str, Any]) -> str | None:
        """Validate parameters against the schema.

        Returns None if valid, error message string if invalid.

        Hint: Check three things:
        1. All required params are present
        2. No unknown params
        3. Enum values are in the allowed set
        """
        raise NotImplementedError("TODO: implement parameter validation")


# =============================================================================
# Part 2: Concrete Tool Implementations
# =============================================================================

class CalculatorTool(Tool):
    """Evaluates mathematical expressions safely.

    Hint: Use a restricted eval() with only math functions in the namespace.
    Block dangerous keywords like __import__, exec, eval, open.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, pi, e."

    @property
    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("expression", "string", "The math expression to evaluate", required=True)
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("TODO: implement calculator tool")


class WeatherTool(Tool):
    """Simulated weather lookup.

    Hint: Use a dict of city -> weather data. Return ToolResult with success=False
    if the city isn't found.
    """

    WEATHER_DATA: dict[str, dict[str, Any]] = {
        "new york": {"temp_f": 72, "condition": "Partly cloudy", "humidity": 65},
        "san francisco": {"temp_f": 58, "condition": "Foggy", "humidity": 80},
        "london": {"temp_f": 55, "condition": "Rainy", "humidity": 85},
        "tokyo": {"temp_f": 68, "condition": "Clear", "humidity": 50},
        "cupertino": {"temp_f": 65, "condition": "Sunny", "humidity": 45},
        "seattle": {"temp_f": 52, "condition": "Overcast", "humidity": 78},
    }

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get the current weather for a city. Returns temperature, conditions, and humidity."

    @property
    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("city", "string", "The city name to look up weather for", required=True)
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("TODO: implement weather tool")


class KnowledgeBaseTool(Tool):
    """Simulated knowledge base search.

    Hint: Use keyword overlap scoring — count words shared between query and each
    knowledge base key. Return the best match.
    """

    KNOWLEDGE: dict[str, str] = {
        "apple headquarters": "Apple Inc. is headquartered in Cupertino, California.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "bitcoin creator": "Bitcoin was created by the pseudonymous Satoshi Nakamoto in 2008.",
        "largest planet": "Jupiter is the largest planet in our solar system.",
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 m/s.",
    }

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return "Search a knowledge base for factual information."

    @property
    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("query", "string", "The search query — use keywords", required=True)
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("TODO: implement knowledge search")


class StringTool(Tool):
    """String manipulation operations.

    Hint: Use an enum parameter to restrict valid operations.
    Map each operation to a simple lambda.
    """

    @property
    def name(self) -> str:
        return "string_tool"

    @property
    def description(self) -> str:
        return "Perform string operations: reverse, uppercase, lowercase, word_count, or char_count."

    @property
    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("text", "string", "The input text to process", required=True),
            ParameterSpec(
                "operation", "string", "The operation to perform",
                required=True,
                enum=["reverse", "uppercase", "lowercase", "word_count", "char_count"]
            ),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("TODO: implement string tool")


# =============================================================================
# Part 3: Tool Registry
# =============================================================================

class ToolRegistry:
    """Central registry for all available tools.

    Hint: Three responsibilities:
    1. Store tools by name (dict) for O(1) lookup
    2. Generate the tool description prompt (iterate tools, format their schemas)
    3. Validate and dispatch tool execution (get tool, validate params, execute)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raise ValueError if name already exists."""
        raise NotImplementedError("TODO: implement tool registration")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name, or None if not found."""
        raise NotImplementedError("TODO: implement tool lookup")

    @property
    def tool_names(self) -> list[str]:
        raise NotImplementedError("TODO: implement tool_names property")

    def generate_tool_prompt(self) -> str:
        """Generate the tool description section for the LLM prompt.

        Hint: For each tool, include name, description, and parameter details.
        End with format instructions for Thought/Action/Action Input.
        """
        raise NotImplementedError("TODO: implement tool prompt generation")

    def execute_tool(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Look up tool, validate params, execute. Return ToolResult."""
        raise NotImplementedError("TODO: implement tool execution dispatch")


# =============================================================================
# Part 4: Action Parsing
# =============================================================================

class ActionType(Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    PARSE_ERROR = "parse_error"


@dataclass
class ParsedAction:
    """Structured representation of the LLM's decision."""
    action_type: ActionType
    thought: str = ""
    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    error: str = ""


def parse_llm_output(output: str) -> ParsedAction:
    """Parse LLM text output into a structured ParsedAction.

    Expected formats:
        Thought: <reasoning>
        Action: <tool_name>
        Action Input: {"param": "value"}

    Or:
        Thought: <reasoning>
        Final Answer: <answer>

    Hint: Use regex to extract each field. Check for Final Answer first.
    Handle malformed output by returning ActionType.PARSE_ERROR with a
    helpful error message.
    """
    raise NotImplementedError("TODO: implement LLM output parsing")


# =============================================================================
# Part 5: Agent Step Tracking
# =============================================================================

@dataclass
class AgentStep:
    """One step in the agent's execution trace."""
    step_number: int
    thought: str
    action_type: str
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = None
    tool_result: str | None = None
    tool_error: str | None = None
    final_answer: str | None = None


# =============================================================================
# Part 6: LLM Simulator
# =============================================================================

class LLMSimulator:
    """Rule-based LLM simulator.

    Hint: Analyze the user query to determine which tool pattern to follow.
    Look at history length to decide if we need another tool call or can
    produce a final answer. Key patterns to implement:
    - Multi-step: "weather at X headquarters" → knowledge_search → weather
    - Calculator: detect math keywords
    - Weather: detect city names
    - Knowledge: detect question words (who, what, where)
    """

    def generate_response(
        self,
        user_query: str,
        tool_prompt: str,
        history: list[AgentStep],
    ) -> str:
        raise NotImplementedError("TODO: implement LLM simulator")


# =============================================================================
# Part 7: The Agent
# =============================================================================

class Agent:
    """The core ReAct agent loop.

    Hint: The loop is:
    1. Call llm.generate_response() with query + history
    2. Parse the output with parse_llm_output()
    3. If FINAL_ANSWER: return it
    4. If TOOL_CALL: execute via registry, append step to history
    5. If PARSE_ERROR: append error step to history
    6. Repeat until max_steps
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm: LLMSimulator,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, user_query: str) -> tuple[str, list[AgentStep]]:
        """Execute the agent loop. Returns (final_answer, step_trace)."""
        raise NotImplementedError("TODO: implement the agent loop")


# =============================================================================
# Test your implementation
# =============================================================================

def create_default_agent(verbose: bool = True) -> Agent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(KnowledgeBaseTool())
    registry.register(StringTool())
    llm = LLMSimulator()
    return Agent(registry, llm, max_steps=10, verbose=verbose)


if __name__ == "__main__":
    agent = create_default_agent(verbose=True)

    print("Test 1: Calculator")
    answer, steps = agent.run("Calculate (17 * 3) + sqrt(144)")
    print(f"Answer: {answer}\n")

    print("Test 2: Multi-step (weather at HQ)")
    answer, steps = agent.run("What's the weather like at Apple's headquarters?")
    print(f"Answer: {answer}\n")

    print("Test 3: Knowledge search")
    answer, steps = agent.run("Who created Python?")
    print(f"Answer: {answer}\n")

    print("Test 4: String tool")
    answer, steps = agent.run('Reverse the text "Hello, Agent World!"')
    print(f"Answer: {answer}\n")
