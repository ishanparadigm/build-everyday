# Day 48: Tool-Using LLM Agent

## Overview

Build an LLM agent that can reason about when and how to use external tools to accomplish tasks. This is the architecture behind every modern AI assistant — from ChatGPT's code interpreter to Claude's tool use. Instead of calling a real LLM API, we'll build the entire agent framework: the tool registry, the reasoning loop, the action parser, and the execution engine. We'll simulate the LLM's decisions to focus on the *systems* side — the part you'd actually build in production.

**Why this matters:** Raw LLMs can only generate text. To make them useful in the real world — querying databases, calling APIs, running code, searching the web — you need an agent loop that lets the model *decide* which tool to call, *parse* its own output into structured actions, *execute* those actions, and *feed results back* for further reasoning. This is the ReAct (Reasoning + Acting) pattern, and understanding it deeply is essential for building production AI systems.

## Core Concepts

### The Agent Loop (ReAct Pattern)

The fundamental insight: an LLM agent is just a **while loop** with an LLM in the middle.

```
while not done:
    observation = format_context(history, tool_results)
    thought + action = llm(observation)       # LLM decides what to do
    if action == "finish":
        done = True
    else:
        result = execute_tool(action)         # We run the tool
        history.append(result)                # Feed result back
```

This is called the **ReAct** pattern (Yao et al., 2022). The key idea is interleaving **reasoning** (the LLM thinking about what to do) with **acting** (executing tools) and **observing** (processing results). Each step builds on previous observations, allowing multi-step problem solving.

**Why not just one LLM call?** Because complex tasks require intermediate results. "What's the weather in the city where Apple is headquartered?" requires: (1) look up Apple's HQ city, (2) get weather for that city. The LLM can't do both in one shot — it needs to see the result of step 1 before deciding step 2.

### Tool Registry and Schema

Tools must be described to the LLM in a structured format so it knows:
- **What tools exist** (name + description)
- **What parameters each tool accepts** (name, type, whether required)
- **What each tool returns**

This is essentially a function signature. The better the description, the better the LLM picks the right tool. In production, this maps directly to OpenAPI specs or JSON Schema.

```python
{
    "name": "calculator",
    "description": "Evaluate a mathematical expression",
    "parameters": {
        "expression": {"type": "string", "description": "Math expression to evaluate", "required": True}
    }
}
```

### Action Parsing

The LLM outputs natural language. We need to extract structured actions from it. There are two main approaches:

1. **Structured output** — Force the LLM to output JSON (what production systems do)
2. **Regex parsing** — Extract tool calls from freeform text (simpler, what we'll implement)

The tradeoff: structured output is more reliable but constrains the LLM's reasoning. Regex parsing is fragile but lets the LLM think freely. Production systems use structured output with a separate "scratchpad" for reasoning.

### Memory and Context Management

As the agent loops, context grows. Each tool result adds tokens. This creates a real engineering problem:
- **Context window limits** — Can't fit infinite history
- **Relevance decay** — Early observations may become irrelevant
- **Cost** — More tokens = more money

Solutions include sliding windows, summarization, and hierarchical memory. We'll implement a simple sliding window with a maximum number of steps.

## Step-by-Step Breakdown

### Step 1: Define the Tool Interface
Create an abstract `Tool` class that all tools must implement. Each tool has a `name`, `description`, `parameters` schema, and an `execute()` method. This is the contract between the agent framework and individual tools.

### Step 2: Build Concrete Tools
Implement several tools: a calculator (eval math expressions), a weather lookup (simulated), a string manipulation tool, and a knowledge base search. Each validates its inputs against its parameter schema before executing.

### Step 3: Create the Tool Registry
A registry that stores tools by name, generates the tool description prompt (what gets sent to the LLM), and dispatches execution requests. This is the "plugin system" of the agent.

### Step 4: Implement the Action Parser
Parse the LLM's output to extract: (1) the thought/reasoning, (2) the chosen tool name, (3) the tool parameters. Handle malformed output gracefully — in production, LLMs frequently produce slightly wrong formats.

### Step 5: Build the Agent Loop
The core ReAct loop: format context -> get LLM response -> parse action -> execute tool -> append to history -> repeat. Include a maximum step limit to prevent infinite loops. Track the full chain of thought for debugging.

### Step 6: Simulate LLM Decisions
Since we're not calling a real API, create a rule-based "LLM simulator" that demonstrates the decision-making process. This lets us test the full framework and understand what the LLM's role is at each step.

### Step 7: Add Observability
Log every step: what the agent thought, what tool it chose, what parameters it used, what result it got. This is critical in production for debugging why an agent made a wrong decision.

## Learning Objectives

- Understand the ReAct agent loop and why it enables multi-step reasoning
- Build a tool registry with schema validation
- Parse structured actions from LLM output
- Implement context management for multi-turn agent conversations
- Design observable, debuggable agent systems
- Connect this to production patterns (LangChain, Claude tool use, OpenAI function calling)

## Going Deeper

- **Real LLM integration**: Replace the simulator with actual Claude API calls using tool_use
- **Parallel tool execution**: Some steps are independent — execute tools concurrently
- **Tool composition**: Let tools call other tools (e.g., a "research" tool that uses search + summarize)
- **Error recovery**: When a tool fails, the agent should retry with different parameters or try a different tool
- **Planning**: Before acting, have the agent create a plan and revise it as results come in (Plan-and-Solve pattern)
- **Evaluation**: Build metrics for agent success rate, tool selection accuracy, and step efficiency
- **Security**: Sandbox tool execution, validate inputs, prevent prompt injection through tool results
