"""
Day 48: Tool-Using LLM Agent

A complete implementation of the ReAct (Reasoning + Acting) agent pattern.
This builds the entire framework that production AI assistants use to call tools:
tool registry, action parsing, execution engine, and the core agent loop.

We simulate LLM decisions with a rule-based system so we can focus on the
systems architecture rather than API calls. In production, you'd swap the
simulator for real Claude/GPT API calls — the framework stays identical.
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

    This mirrors JSON Schema / OpenAPI parameter definitions — the same format
    used by Claude's tool_use and OpenAI's function_calling APIs.
    """
    name: str
    type: str  # "string", "number", "boolean"
    description: str
    required: bool = True
    enum: list[str] | None = None  # Optional list of allowed values


@dataclass
class ToolResult:
    """Result of executing a tool.

    We wrap results in a structured object rather than returning raw strings
    because we need to distinguish success from failure, and attach metadata
    for the agent's context management.
    """
    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    """Abstract base class for all tools.

    Every tool in the registry must implement this interface. The key design
    decision: tools are self-describing. They carry their own name, description,
    and parameter schema. This means the agent framework doesn't need to know
    anything about specific tools — it just reads the schema and dispatches.

    This is the Plugin pattern: new tools can be added without modifying the
    agent or registry code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description. This is what the LLM reads to decide
        whether to use this tool. Quality of this description directly affects
        tool selection accuracy."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ParameterSpec]:
        """Parameter schema. The agent uses this to validate tool calls
        before execution, catching errors early."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given parameters.

        This is where the actual work happens. In production, this might call
        an API, query a database, or run code in a sandbox.
        """
        ...

    def validate_params(self, params: dict[str, Any]) -> str | None:
        """Validate parameters against the schema.

        Returns None if valid, or an error message string if invalid.
        We validate BEFORE execution to give the LLM clear feedback about
        what went wrong, so it can self-correct on the next step.
        """
        param_specs = {p.name: p for p in self.parameters}

        # Check required parameters are present
        for spec in self.parameters:
            if spec.required and spec.name not in params:
                return f"Missing required parameter: '{spec.name}'"

        # Check no unknown parameters
        for key in params:
            if key not in param_specs:
                return f"Unknown parameter: '{key}'. Valid parameters: {list(param_specs.keys())}"

        # Check enum constraints
        for key, value in params.items():
            spec = param_specs[key]
            if spec.enum and value not in spec.enum:
                return f"Invalid value '{value}' for '{key}'. Must be one of: {spec.enum}"

        return None


# =============================================================================
# Part 2: Concrete Tool Implementations
# =============================================================================

class CalculatorTool(Tool):
    """Evaluates mathematical expressions safely.

    In production, you'd use a sandboxed code execution environment.
    Here we use a restricted eval with only math operations allowed.
    NEVER use raw eval() in production — it's a code injection vector.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, pi, e. Example: '(3 + 4) * 2' or 'sqrt(144)'"

    @property
    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("expression", "string", "The math expression to evaluate", required=True)
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        expr = kwargs["expression"]

        # Restricted namespace — only math functions, no builtins.
        # This prevents code injection while allowing useful math.
        safe_dict: dict[str, Any] = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs, "pow": pow,
            "round": round, "min": min, "max": max,
        }
        # Block dunder attributes which could be used to escape the sandbox
        if any(kw in expr for kw in ["__", "import", "exec", "eval", "open"]):
            return ToolResult(success=False, output="", error="Expression contains forbidden keywords")

        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Math error: {e}")


class WeatherTool(Tool):
    """Simulated weather lookup.

    In production this would call a weather API. We simulate it to demonstrate
    the tool interface without requiring API keys. The agent framework treats
    this identically to a real API tool.
    """

    # Simulated weather data
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
        city = kwargs["city"].lower().strip()
        if city in self.WEATHER_DATA:
            data = self.WEATHER_DATA[city]
            return ToolResult(
                success=True,
                output=f"{city.title()}: {data['temp_f']}°F, {data['condition']}, {data['humidity']}% humidity"
            )
        return ToolResult(
            success=False, output="",
            error=f"No weather data for '{city}'. Available cities: {', '.join(self.WEATHER_DATA.keys())}"
        )


class KnowledgeBaseTool(Tool):
    """Simulated knowledge base search.

    Represents a retrieval system — could be a vector DB, search index, or
    structured database. The agent uses this when it needs factual information
    it doesn't have in context.
    """

    KNOWLEDGE: dict[str, str] = {
        "apple headquarters": "Apple Inc. is headquartered in Cupertino, California.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "bitcoin creator": "Bitcoin was created by the pseudonymous Satoshi Nakamoto in 2008.",
        "largest planet": "Jupiter is the largest planet in our solar system with a mass of 1.898 × 10^27 kg.",
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "transformer architecture": "The Transformer was introduced in 'Attention Is All You Need' (Vaswani et al., 2017). It uses self-attention instead of recurrence.",
        "react pattern": "ReAct (Reasoning + Acting) interleaves LLM reasoning with tool execution for multi-step problem solving.",
        "population earth": "Earth's population is approximately 8 billion people as of 2024.",
    }

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return "Search a knowledge base for factual information. Use this when you need to look up facts, definitions, or reference data."

    @property
    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec("query", "string", "The search query — use keywords, not full questions", required=True)
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs["query"].lower().strip()

        # Simple keyword matching — in production you'd use embeddings/vector search
        # (which we built in Day 47!)
        best_match = None
        best_score = 0

        for key, value in self.KNOWLEDGE.items():
            # Count overlapping words between query and key
            query_words = set(query.split())
            key_words = set(key.split())
            score = len(query_words & key_words)
            if score > best_score:
                best_score = score
                best_match = value

        if best_match and best_score > 0:
            return ToolResult(success=True, output=best_match)
        return ToolResult(
            success=False, output="",
            error=f"No relevant results for '{query}'"
        )


class StringTool(Tool):
    """String manipulation operations.

    Demonstrates a tool with an enum parameter — the LLM must choose from
    a fixed set of operations. This pattern is common in production for
    tools with multiple modes.
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
        text = kwargs["text"]
        op = kwargs["operation"]

        operations = {
            "reverse": lambda t: t[::-1],
            "uppercase": lambda t: t.upper(),
            "lowercase": lambda t: t.lower(),
            "word_count": lambda t: str(len(t.split())),
            "char_count": lambda t: str(len(t)),
        }

        result = operations[op](text)
        return ToolResult(success=True, output=result)


# =============================================================================
# Part 3: Tool Registry
# =============================================================================

class ToolRegistry:
    """Central registry for all available tools.

    Responsibilities:
    1. Store tools by name for O(1) lookup
    2. Generate the tool description prompt (what the LLM sees)
    3. Validate and dispatch tool execution requests

    This is the "plugin manager" of the agent system. In production frameworks
    like LangChain, this is the ToolManager. In Claude's API, this maps to
    the 'tools' parameter in the API request.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if name conflicts — tool names must be unique
        because the LLM references tools by name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def generate_tool_prompt(self) -> str:
        """Generate the tool description section of the system prompt.

        This is what the LLM reads to understand what tools are available.
        The format must be clear and consistent — the LLM's tool selection
        accuracy depends directly on the quality of these descriptions.
        """
        lines = ["You have access to the following tools:\n"]

        for tool in self._tools.values():
            lines.append(f"## {tool.name}")
            lines.append(f"Description: {tool.description}")
            lines.append("Parameters:")
            for param in tool.parameters:
                req = "REQUIRED" if param.required else "optional"
                enum_str = f" (one of: {param.enum})" if param.enum else ""
                lines.append(f"  - {param.name} ({param.type}, {req}): {param.description}{enum_str}")
            lines.append("")

        lines.append("To use a tool, respond with:")
        lines.append("Thought: <your reasoning>")
        lines.append("Action: <tool_name>")
        lines.append('Action Input: {"param": "value"}')
        lines.append("")
        lines.append("To give a final answer:")
        lines.append("Thought: <your reasoning>")
        lines.append("Final Answer: <your answer>")

        return "\n".join(lines)

    def execute_tool(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Validate and execute a tool call.

        This is the single entry point for all tool execution. Having one
        entry point means we can add logging, rate limiting, sandboxing,
        and access control in one place.
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False, output="",
                error=f"Unknown tool: '{name}'. Available tools: {self.tool_names}"
            )

        # Validate parameters before execution
        validation_error = tool.validate_params(params)
        if validation_error:
            return ToolResult(success=False, output="", error=validation_error)

        return tool.execute(**params)


# =============================================================================
# Part 4: Action Parsing
# =============================================================================

class ActionType(Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    PARSE_ERROR = "parse_error"


@dataclass
class ParsedAction:
    """Structured representation of what the LLM decided to do.

    Every LLM response is parsed into one of three types:
    - TOOL_CALL: The LLM wants to use a tool (has tool_name + params)
    - FINAL_ANSWER: The LLM is done and has an answer
    - PARSE_ERROR: We couldn't understand the LLM's output

    Parse errors are critical to handle well — they happen frequently in
    production. The agent should feed the error back so the LLM can
    self-correct its output format.
    """
    action_type: ActionType
    thought: str = ""
    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    error: str = ""


def parse_llm_output(output: str) -> ParsedAction:
    """Parse the LLM's text output into a structured action.

    We use regex to extract the Thought/Action/Action Input pattern.
    This is inherently fragile — LLMs don't always follow the format exactly.
    Production systems use structured output (JSON mode) to avoid this.

    But understanding regex parsing is valuable because:
    1. It's how early agent systems (AutoGPT, BabyAGI) worked
    2. It teaches you what can go wrong and why structured output matters
    3. Some models don't support structured output
    """
    # Extract thought (reasoning trace)
    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)", output, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    # Check for final answer first
    final_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
    if final_match:
        return ParsedAction(
            action_type=ActionType.FINAL_ANSWER,
            thought=thought,
            final_answer=final_match.group(1).strip()
        )

    # Try to parse tool call
    action_match = re.search(r"Action:\s*(\w+)", output)
    if not action_match:
        return ParsedAction(
            action_type=ActionType.PARSE_ERROR,
            thought=thought,
            error="Could not find 'Action: <tool_name>' in response. Please use the correct format."
        )

    tool_name = action_match.group(1).strip()

    # Parse Action Input as JSON
    input_match = re.search(r"Action Input:\s*(\{.+?\})", output, re.DOTALL)
    if not input_match:
        return ParsedAction(
            action_type=ActionType.PARSE_ERROR,
            thought=thought,
            error="Could not find 'Action Input: {\"param\": \"value\"}' in response."
        )

    try:
        params = json.loads(input_match.group(1))
    except json.JSONDecodeError as e:
        return ParsedAction(
            action_type=ActionType.PARSE_ERROR,
            thought=thought,
            error=f"Invalid JSON in Action Input: {e}"
        )

    return ParsedAction(
        action_type=ActionType.TOOL_CALL,
        thought=thought,
        tool_name=tool_name,
        tool_params=params
    )


# =============================================================================
# Part 5: Agent Step Tracking
# =============================================================================

@dataclass
class AgentStep:
    """One step in the agent's execution trace.

    Every step records what the agent thought, what it did, and what happened.
    This is the fundamental unit of observability — when an agent makes a wrong
    decision, you debug by examining the step trace.
    """
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
    """Rule-based LLM simulator for testing the agent framework.

    This replaces the actual LLM API call. It analyzes the user query and
    conversation history to produce realistic Thought/Action/Action Input
    responses. The rules demonstrate common agent reasoning patterns:

    1. Multi-step: "weather in Apple's HQ" → search HQ → get weather
    2. Direct tool use: "what's 2+2" → calculator
    3. Information synthesis: combine results from multiple tool calls

    In production, you'd replace generate_response() with an API call to
    Claude or GPT. Everything else in the agent stays the same.
    """

    def generate_response(
        self,
        user_query: str,
        tool_prompt: str,
        history: list[AgentStep],
    ) -> str:
        """Simulate an LLM response given the current context.

        The simulator looks at:
        - The original user query (what we're trying to accomplish)
        - Previous steps (what we've already done and learned)
        - Tool results (what information we now have)

        Then it decides: do we need another tool call, or can we answer?
        """
        query_lower = user_query.lower()

        # === Pattern: Multi-step reasoning (chained tool calls) ===
        # "Weather in the city where X is headquartered" requires two steps
        if "weather" in query_lower and "headquarter" in query_lower:
            if not history:
                # Step 1: First, look up the headquarters
                company = "apple"  # Default; a real LLM would extract this
                for word in ["apple", "google", "microsoft", "meta"]:
                    if word in query_lower:
                        company = word
                        break
                return (
                    f"Thought: The user wants weather for {company}'s headquarters city. "
                    f"I first need to find where {company} is headquartered.\n"
                    f"Action: knowledge_search\n"
                    f'Action Input: {{"query": "{company} headquarters"}}'
                )
            elif len(history) == 1 and history[0].tool_result:
                # Step 2: Extract city from knowledge result, then get weather
                result = history[0].tool_result
                # Try to extract a city name from the result
                city = "cupertino"  # Default
                for c in ["cupertino", "mountain view", "redmond", "menlo park", "new york", "san francisco", "seattle"]:
                    if c in result.lower():
                        city = c
                        break
                return (
                    f"Thought: I found that the headquarters is in {city.title()}. "
                    f"Now I need to get the weather there.\n"
                    f"Action: weather\n"
                    f'Action Input: {{"city": "{city}"}}'
                )
            else:
                # Step 3: We have both pieces of info, synthesize final answer
                hq_info = history[0].tool_result or "unknown location"
                weather_info = history[-1].tool_result or "unknown weather"
                return (
                    f"Thought: I now have both the headquarters location and the weather. "
                    f"Let me combine these into a final answer.\n"
                    f"Final Answer: {hq_info} The current weather there: {weather_info}"
                )

        # === Pattern: Direct calculator use ===
        if any(op in query_lower for op in ["calculate", "compute", "what is", "solve", "math"]):
            if not history:
                # Extract the math expression
                expr = query_lower
                for prefix in ["calculate ", "compute ", "what is ", "solve ", "what's "]:
                    if prefix in expr:
                        expr = expr.split(prefix, 1)[1].strip().rstrip("?")
                        break
                return (
                    f"Thought: The user wants me to calculate a math expression. "
                    f"I'll use the calculator tool.\n"
                    f"Action: calculator\n"
                    f'Action Input: {{"expression": "{expr}"}}'
                )
            else:
                result = history[-1].tool_result or "error"
                return (
                    f"Thought: The calculator returned the result. I can now answer.\n"
                    f"Final Answer: The result is {result}"
                )

        # === Pattern: Weather lookup ===
        if "weather" in query_lower:
            if not history:
                city = "new york"  # default
                for c in ["new york", "san francisco", "london", "tokyo", "cupertino", "seattle"]:
                    if c in query_lower:
                        city = c
                        break
                return (
                    f"Thought: The user wants to know the weather. "
                    f"I'll look up the weather for {city.title()}.\n"
                    f"Action: weather\n"
                    f'Action Input: {{"city": "{city}"}}'
                )
            else:
                result = history[-1].tool_result or "unknown"
                return (
                    f"Thought: I have the weather data.\n"
                    f"Final Answer: {result}"
                )

        # === Pattern: String operations ===
        if any(op in query_lower for op in ["reverse", "uppercase", "lowercase", "word count", "count words", "count characters"]):
            if not history:
                # Determine operation
                if "reverse" in query_lower:
                    operation = "reverse"
                elif "uppercase" in query_lower or "upper case" in query_lower:
                    operation = "uppercase"
                elif "lowercase" in query_lower or "lower case" in query_lower:
                    operation = "lowercase"
                elif "word" in query_lower and "count" in query_lower:
                    operation = "word_count"
                else:
                    operation = "char_count"

                # Extract text (between quotes or after "of"/"the text")
                text_match = re.search(r'"([^"]+)"', user_query)
                if text_match:
                    text = text_match.group(1)
                else:
                    text = "hello world"

                return (
                    f"Thought: The user wants to {operation} some text. "
                    f"I'll use the string tool.\n"
                    f"Action: string_tool\n"
                    f'Action Input: {{"text": "{text}", "operation": "{operation}"}}'
                )
            else:
                result = history[-1].tool_result or "error"
                return (
                    f"Thought: Got the result from the string tool.\n"
                    f"Final Answer: The result is: {result}"
                )

        # === Pattern: Knowledge lookup ===
        if any(kw in query_lower for kw in ["who", "what", "where", "tell me about", "look up"]):
            if not history:
                # Extract the query from the user's question
                query = query_lower
                for prefix in ["who is the ", "who created ", "what is the ", "where is ", "tell me about ", "look up "]:
                    if prefix in query:
                        query = query.split(prefix, 1)[1].strip().rstrip("?")
                        break
                return (
                    f"Thought: The user is asking a factual question. "
                    f"Let me search the knowledge base.\n"
                    f"Action: knowledge_search\n"
                    f'Action Input: {{"query": "{query}"}}'
                )
            else:
                result = history[-1].tool_result or "I couldn't find that information"
                return (
                    f"Thought: Found the answer in the knowledge base.\n"
                    f"Final Answer: {result}"
                )

        # === Fallback: Direct answer ===
        return (
            "Thought: I can answer this directly without using any tools.\n"
            "Final Answer: I'm not sure how to help with that. "
            "I can calculate math, look up weather, search a knowledge base, or manipulate strings."
        )


# =============================================================================
# Part 7: The Agent
# =============================================================================

class Agent:
    """The core ReAct agent.

    This ties everything together: the tool registry, the LLM (simulator),
    the action parser, and the execution loop. The agent's job is simple:

    1. Ask the LLM what to do
    2. Parse the response
    3. If it's a tool call: execute the tool, record the result, go to 1
    4. If it's a final answer: we're done
    5. If it's a parse error: tell the LLM to fix its format, go to 1

    Max steps prevents infinite loops (a real risk with LLMs that get confused
    and keep calling the same tool repeatedly).
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
        """Execute the agent loop for a user query.

        Returns:
            tuple of (final_answer, step_trace)

        The step trace is the full execution history — essential for
        debugging and evaluation. In production, you'd log this to
        an observability platform.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  AGENT QUERY: {user_query}")
            print(f"{'='*60}")

        tool_prompt = self.registry.generate_tool_prompt()
        history: list[AgentStep] = []

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step_num} ---")

            # 1. Get LLM response (in production: API call to Claude/GPT)
            response = self.llm.generate_response(user_query, tool_prompt, history)

            if self.verbose:
                print(f"LLM Output:\n{response}")

            # 2. Parse the response into a structured action
            parsed = parse_llm_output(response)

            # 3. Handle based on action type
            if parsed.action_type == ActionType.FINAL_ANSWER:
                step = AgentStep(
                    step_number=step_num,
                    thought=parsed.thought,
                    action_type="final_answer",
                    final_answer=parsed.final_answer,
                )
                history.append(step)

                if self.verbose:
                    print(f"\n  FINAL ANSWER: {parsed.final_answer}")

                return parsed.final_answer, history

            elif parsed.action_type == ActionType.TOOL_CALL:
                # Execute the tool
                if self.verbose:
                    print(f"  -> Calling tool: {parsed.tool_name}({parsed.tool_params})")

                result = self.registry.execute_tool(parsed.tool_name, parsed.tool_params)

                step = AgentStep(
                    step_number=step_num,
                    thought=parsed.thought,
                    action_type="tool_call",
                    tool_name=parsed.tool_name,
                    tool_params=parsed.tool_params,
                    tool_result=result.output if result.success else None,
                    tool_error=result.error,
                )
                history.append(step)

                if self.verbose:
                    if result.success:
                        print(f"  <- Tool result: {result.output}")
                    else:
                        print(f"  <- Tool error: {result.error}")

            elif parsed.action_type == ActionType.PARSE_ERROR:
                # Feed the error back so the LLM can self-correct
                step = AgentStep(
                    step_number=step_num,
                    thought=parsed.thought,
                    action_type="parse_error",
                    tool_error=parsed.error,
                )
                history.append(step)

                if self.verbose:
                    print(f"  !! Parse error: {parsed.error}")

        # If we hit max steps, return what we have
        if self.verbose:
            print(f"\n  !! Max steps ({self.max_steps}) reached without final answer")

        return "I wasn't able to complete this task within the step limit.", history


# =============================================================================
# Part 8: Demo and Main
# =============================================================================

def create_default_agent(verbose: bool = True) -> Agent:
    """Factory function to create an agent with all default tools registered."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(KnowledgeBaseTool())
    registry.register(StringTool())

    llm = LLMSimulator()
    return Agent(registry, llm, max_steps=10, verbose=verbose)


def print_step_trace(steps: list[AgentStep]) -> None:
    """Pretty-print the agent's execution trace."""
    print(f"\n{'='*60}")
    print("  EXECUTION TRACE")
    print(f"{'='*60}")
    for step in steps:
        print(f"\n  Step {step.step_number}: [{step.action_type}]")
        if step.thought:
            print(f"    Thought: {step.thought[:80]}...")
        if step.tool_name:
            print(f"    Tool: {step.tool_name}({step.tool_params})")
        if step.tool_result:
            print(f"    Result: {step.tool_result}")
        if step.tool_error:
            print(f"    Error: {step.tool_error}")
        if step.final_answer:
            print(f"    Answer: {step.final_answer}")


if __name__ == "__main__":
    agent = create_default_agent(verbose=True)

    # --- Demo 1: Single-step tool use (calculator) ---
    print("\n" + "=" * 70)
    print(" DEMO 1: Direct calculator use")
    print("=" * 70)
    answer, steps = agent.run("Calculate (17 * 3) + sqrt(144)")
    print_step_trace(steps)

    # --- Demo 2: Multi-step reasoning (chained tool calls) ---
    print("\n\n" + "=" * 70)
    print(" DEMO 2: Multi-step reasoning — weather at Apple's headquarters")
    print("=" * 70)
    answer, steps = agent.run("What's the weather like at Apple's headquarters?")
    print_step_trace(steps)

    # --- Demo 3: Knowledge base search ---
    print("\n\n" + "=" * 70)
    print(" DEMO 3: Knowledge base lookup")
    print("=" * 70)
    answer, steps = agent.run("Who created Python?")
    print_step_trace(steps)

    # --- Demo 4: String manipulation ---
    print("\n\n" + "=" * 70)
    print(" DEMO 4: String tool with enum parameter")
    print("=" * 70)
    answer, steps = agent.run('Reverse the text "Hello, Agent World!"')
    print_step_trace(steps)

    # --- Demo 5: Weather lookup ---
    print("\n\n" + "=" * 70)
    print(" DEMO 5: Direct weather lookup")
    print("=" * 70)
    answer, steps = agent.run("What's the weather in Tokyo?")
    print_step_trace(steps)

    # --- Show the tool prompt (what the LLM sees) ---
    print("\n\n" + "=" * 70)
    print(" APPENDIX: Tool description prompt sent to the LLM")
    print("=" * 70)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(KnowledgeBaseTool())
    registry.register(StringTool())
    print(registry.generate_tool_prompt())

    print("\n\nAll demos completed successfully!")
