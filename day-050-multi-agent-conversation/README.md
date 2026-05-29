# Day 50: Multi-Agent Conversation System

## Overview

Build a multi-agent conversation system where multiple AI agents with distinct roles collaborate to solve problems through structured dialogue. This is the backbone of modern AI orchestration — from code review pipelines (where a planner, coder, and reviewer each play a part) to autonomous research systems where agents debate, refine, and synthesize answers.

Unlike single-agent systems (Day 48), multi-agent architectures introduce **coordination**, **role specialization**, and **conversation management** — the same challenges that distributed systems face, but at the reasoning layer.

## Core Concepts

### 1. Agent Roles and Specialization

Each agent is defined by a **system prompt** that constrains its behavior, a **name/identity**, and a set of **capabilities**. The key insight is that *narrower roles produce better outputs* — a "code reviewer" agent that only critiques code outperforms a general agent asked to "also review the code."

Why? LLMs respond strongly to role framing. When you tell a model "you are a security auditor," it activates different attention patterns than "you are a helpful assistant." Specialization is essentially prompt engineering at the agent level.

### 2. Conversation Topology

How agents talk to each other matters enormously:

- **Round-robin**: Each agent speaks in turn. Simple, predictable, but can be wasteful (an agent speaks even when it has nothing to add).
- **Hub-and-spoke**: A coordinator agent decides who speaks next. More efficient, but the coordinator becomes a bottleneck and single point of failure.
- **Broadcast**: Every agent sees every message. High bandwidth but scales poorly — N agents means N evaluations per message.
- **Directed**: Agents address specific other agents. Most flexible, closest to how human teams work.

The topology you choose affects latency, cost, and quality. Round-robin is O(N) per round; broadcast is O(N^2).

### 3. Conversation State and Memory

Multi-agent systems need shared state:

- **Message history**: The full conversation transcript, visible to all or selectively filtered per agent.
- **Shared context**: Documents, code, data that agents reference.
- **Agent state**: Each agent's internal beliefs, accumulated findings, or working memory.

The critical design decision: **full history vs. summarized context**. Full history is accurate but expensive (token costs scale linearly). Summarized context is cheaper but lossy — agents may miss nuances.

### 4. Termination Conditions

Without explicit stopping criteria, agents can loop forever (Agent A asks Agent B, who asks Agent A...). Common strategies:

- **Max rounds**: Hard cap on conversation turns. Simple but may cut off productive dialogue.
- **Consensus detection**: Stop when agents agree. Requires a way to detect agreement (e.g., a voting mechanism or explicit "I agree" signals).
- **Coordinator decision**: The hub agent decides when the goal is met.
- **Quality threshold**: Stop when a metric (e.g., confidence score, test pass rate) exceeds a threshold.

### 5. The Orchestrator Pattern

The most practical multi-agent pattern in production:

```
User Query → Orchestrator → [Agent A, Agent B, Agent C] → Orchestrator → Response
```

The orchestrator:
1. Receives the user's request
2. Decomposes it into subtasks
3. Routes subtasks to specialized agents
4. Synthesizes their responses
5. Decides if more rounds are needed

This is essentially a **dispatch loop** — the same pattern as an OS scheduler or a message broker.

## Step-by-Step Breakdown

### Step 1: Define the Agent abstraction
Create an `Agent` class that encapsulates a name, role description (system prompt), and a method to generate responses given a conversation history. Each agent maintains its own perspective but operates on shared message history.

### Step 2: Build the message protocol
Define a `Message` dataclass with sender, content, and timestamp. This is the unit of communication — every agent reads and writes messages in the same format. Without a standard protocol, agents can't interoperate.

### Step 3: Implement the conversation manager
The `ConversationManager` maintains the message list, selects which agent speaks next (based on topology), and checks termination conditions. It's the runtime that drives the multi-agent loop.

### Step 4: Implement conversation topologies
Build at least round-robin and coordinator-based selection. The coordinator topology requires one designated agent that outputs the name of the next speaker.

### Step 5: Add termination logic
Implement max-rounds, consensus detection (keyword-based), and coordinator-decision stopping. Multiple conditions can be combined (e.g., stop on consensus OR after 10 rounds).

### Step 6: Build a practical example
Create a "code review" pipeline: a Developer agent writes code, a Reviewer agent critiques it, and an Architect agent makes final decisions. Run them through a multi-round conversation to refine a solution.

## Learning Objectives

- Design agent abstractions with role specialization
- Implement multiple conversation topologies (round-robin, coordinator)
- Build conversation state management with shared message history
- Handle termination conditions to prevent infinite loops
- Understand the tradeoffs between different multi-agent architectures
- See how multi-agent patterns connect to distributed systems concepts

## Going Deeper

- **Debate as alignment**: Anthropic and others have explored using multi-agent debate to improve truthfulness — agents arguing opposing sides surface more nuanced answers than a single agent.
- **AutoGen / CrewAI**: Production frameworks that implement these patterns. Study their agent-to-agent protocols.
- **Cost optimization**: In production, you'd use cheaper models for routine agents (summarizer, formatter) and expensive models only for the "thinker" agents.
- **Tool-augmented agents**: Combine Day 48's tool-using agents with today's multi-agent system — each agent gets different tools (one can search the web, another can run code).
- **Async execution**: In production, agents run in parallel where possible. The orchestrator dispatches independent subtasks concurrently, only serializing when there are dependencies.
