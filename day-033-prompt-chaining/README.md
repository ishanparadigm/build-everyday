# Day 033: Prompt Chaining with Claude API

## Overview

Build a **multi-step prompt chaining pipeline** that decomposes complex tasks into sequential LLM calls, where each step's output feeds into the next step's prompt. This is one of the most practical patterns in production AI systems — it turns unreliable, monolithic prompts into reliable, debuggable, composable pipelines.

**Why it matters:** A single LLM call trying to do everything (analyze, reason, decide, format) often produces mediocre results. Prompt chaining breaks this into specialized steps, each with a focused system prompt and clear input/output contract. This is how production systems like code review bots, content pipelines, and autonomous agents actually work.

## Core Concepts

### 1. The Composability Principle

A chain of N specialized prompts almost always outperforms a single prompt trying to do N things at once. Why?

- **Cognitive load:** LLMs degrade when instructions are complex. A prompt doing one thing well is more reliable than a prompt juggling five concerns.
- **Debuggability:** When a chain fails, you can inspect the intermediate outputs and identify exactly which step broke. With a monolithic prompt, you're guessing.
- **Reusability:** Individual steps can be mixed and matched across different pipelines.

The tradeoff is latency and cost — each step is a separate API call. The art is finding the right granularity: too few steps and you lose reliability, too many and you pay unnecessary latency/cost.

### 2. Chain Architecture Patterns

**Sequential Chain:** A → B → C. Each step depends on the previous. Best for pipelines where each step transforms or enriches the data.

```
[Raw Input] → [Step 1: Extract] → [Step 2: Analyze] → [Step 3: Format] → [Final Output]
```

**Gate Chain:** A step decides whether to continue, branch, or abort. Adds conditional logic to the pipeline.

```
[Input] → [Classify] → if "complex" → [Deep Analysis] → [Output]
                      → if "simple" → [Quick Response] → [Output]
```

**Validation Chain:** A step checks the previous step's output and either passes it through or sends it back for retry.

```
[Generate] → [Validate] → if valid → [Output]
                         → if invalid → [Generate] (retry with feedback)
```

### 3. Prompt Design for Chains

Each step in a chain needs:
- **A system prompt** defining its role and output format
- **Structured input** from the previous step (usually the previous step's output, possibly reformatted)
- **A clear output contract** so the next step knows what to expect

The key insight: **the output format of step N must match the expected input format of step N+1**. This is where most chains break — format mismatches between steps.

### 4. Error Handling and Retries

LLM outputs are stochastic. A chain must handle:
- **Malformed outputs:** A step returns text that doesn't match the expected format
- **Hallucinations:** A step invents facts not present in the input
- **Refusals:** The model declines to process certain content
- **Rate limits / API errors:** Transient failures from the API

Production chains use validation functions between steps that parse and verify outputs before passing them forward.

### 5. Context Window Management

Each step in the chain gets a fresh context window. This is both a feature and a constraint:
- **Feature:** Each step can have a focused, optimized system prompt without leftover context
- **Constraint:** Information from early steps must be explicitly passed forward — the model doesn't "remember" previous steps

The design question: should you pass the full accumulated context to each step, or only the minimal information each step needs? Passing everything is safer but wastes tokens; passing minimal context is efficient but risks losing information.

## Step-by-Step Breakdown

### Step 1: Define the Chain Structure

We'll build a **research and analysis pipeline** that takes a topic and produces a structured analysis:

1. **Research Planner** — takes a topic, produces 3-5 specific research questions
2. **Researcher** — takes each question, produces key findings (runs in parallel)
3. **Synthesizer** — takes all findings, produces a coherent analysis
4. **Critic** — reviews the analysis for gaps and unsupported claims
5. **Final Editor** — produces the polished output incorporating the critic's feedback

This is a real pattern used in AI writing assistants and research tools.

### Step 2: Implement the Chain Runner

The chain runner orchestrates step execution:
- Manages the API client
- Passes outputs between steps
- Handles errors and retries
- Logs intermediate results for debugging
- Tracks token usage and latency per step

### Step 3: Build Validation Between Steps

Between each step, a validation function checks:
- Did the output parse correctly?
- Does it contain the required fields?
- Is it within expected length bounds?

If validation fails, the step retries with feedback about what went wrong.

### Step 4: Add Parallel Execution

Some steps can run in parallel (e.g., researching multiple questions simultaneously). We use `asyncio` to run independent sub-tasks concurrently, reducing total pipeline latency.

### Step 5: Instrument the Pipeline

Add logging and metrics so you can:
- See each step's input/output
- Track token usage per step
- Measure latency per step
- Identify which steps fail most often

## Learning Objectives

- Understand prompt chaining as an architectural pattern for LLM applications
- Design multi-step pipelines with clear input/output contracts between steps
- Implement validation and retry logic for stochastic LLM outputs
- Use async/await for parallel LLM calls
- Build observable pipelines with per-step logging and metrics
- Manage context passing between chain steps efficiently

## Going Deeper

- **Streaming chains:** Instead of waiting for each step to complete, stream tokens from one step into the next. This reduces perceived latency for user-facing applications.
- **Dynamic chains:** Steps that decide at runtime which step to execute next, enabling complex branching logic (this is how agents work under the hood).
- **Caching:** Memoize step outputs keyed on input hash. If the same sub-question appears across multiple runs, skip the API call.
- **Evaluation:** Build an eval harness that runs the same input through your chain 10 times and measures output consistency. High variance means your chain is fragile.
- **Cost optimization:** Profile which steps consume the most tokens and consider using smaller/cheaper models for simple steps (e.g., formatting) while reserving expensive models for reasoning-heavy steps.
- **Connection to agents:** An agent is essentially a dynamic chain where one step (the "planning" step) decides which tool or sub-chain to invoke next. Understanding static chains is prerequisite to building agents (Day 036 will cover this).
