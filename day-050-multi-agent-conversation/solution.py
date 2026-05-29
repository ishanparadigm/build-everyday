"""
Day 50: Multi-Agent Conversation System

A framework for orchestrating conversations between multiple AI agents with
distinct roles. Demonstrates round-robin and coordinator topologies, shared
message history, and termination conditions.

Since we want this to run without API keys, agents use rule-based response
generation that simulates LLM behavior. The architecture is identical to what
you'd build with real LLM calls — just swap the generate() method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol
import re
import textwrap


# =============================================================================
# Message Protocol
# =============================================================================
# Every agent reads and writes the same message format. This is the "wire
# protocol" of our multi-agent system — without it, agents can't interoperate.

@dataclass
class Message:
    """A single message in the conversation."""
    sender: str          # Agent name or "user" or "system"
    content: str         # The message text
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)  # For extensibility (e.g., confidence scores)

    def __str__(self) -> str:
        return f"[{self.sender}]: {self.content}"


# =============================================================================
# Agent Protocol and Base Implementation
# =============================================================================
# We use a Protocol (structural typing) so any object with a respond() method
# can act as an agent. This is more Pythonic than forcing inheritance.

class AgentProtocol(Protocol):
    """Structural type for anything that can act as an agent."""
    name: str
    role: str
    def respond(self, history: list[Message]) -> Optional[str]: ...


class Agent:
    """
    An agent with a defined role that generates responses based on conversation
    history. In production, the respond() method would call an LLM API.

    The role description acts as a "system prompt" — it constrains what the
    agent focuses on and how it responds. Narrow roles produce better outputs
    because they reduce the agent's decision space.
    """

    def __init__(self, name: str, role: str, behaviors: dict[str, str] | None = None):
        """
        Args:
            name: Agent identifier (e.g., "Reviewer", "Architect")
            role: Role description that guides the agent's responses
            behaviors: Dict mapping trigger keywords to response templates.
                       This simulates LLM behavior without API calls.
        """
        self.name = name
        self.role = role
        # Behaviors map keyword patterns to response templates.
        # In a real system, the LLM's system prompt replaces this entirely.
        self.behaviors: dict[str, str] = behaviors or {}
        self._response_count = 0  # Track how many times this agent has spoken
        self._used_behaviors: set[str] = set()  # Track used triggers to avoid repetition

    def respond(self, history: list[Message]) -> Optional[str]:
        """
        Generate a response given the conversation history.

        The agent scans recent messages for trigger keywords and responds
        accordingly. If no triggers match, it provides a default response
        based on its role.

        Returns None if the agent has nothing to add (important for efficient
        conversations — not every agent needs to speak every round).
        """
        if not history:
            return f"As the {self.role}, I'm ready to contribute."

        # Don't respond to yourself — prevents echo loops
        last_msg = history[-1]
        if last_msg.sender == self.name:
            return None

        # Look at the last few messages for context
        recent = history[-3:]
        recent_text = " ".join(m.content.lower() for m in recent)

        # Check behavior triggers (keyword → response mapping)
        # Only fire each trigger once to prevent repetitive loops
        for trigger, response in self.behaviors.items():
            if trigger.lower() in recent_text and trigger not in self._used_behaviors:
                self._used_behaviors.add(trigger)
                self._response_count += 1
                return response

        # Default: acknowledge and contribute based on role
        self._response_count += 1
        return f"From my perspective as {self.role}: I've reviewed the discussion and have input on {last_msg.sender}'s points."

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r})"


# =============================================================================
# Conversation Topologies
# =============================================================================
# The topology determines WHO speaks WHEN. This is the scheduling algorithm
# of multi-agent systems — analogous to process scheduling in an OS.

class Topology(Enum):
    ROUND_ROBIN = "round_robin"      # Each agent speaks in turn
    COORDINATOR = "coordinator"       # A designated agent picks who speaks next
    BROADCAST = "broadcast"           # All agents respond to each message


class AgentSelector:
    """
    Selects the next agent to speak based on the conversation topology.

    Why a separate class? Because selection logic is orthogonal to conversation
    management. You might want to swap topologies mid-conversation (e.g., switch
    from round-robin to coordinator when agents can't agree).
    """

    def __init__(self, agents: list[Agent], topology: Topology,
                 coordinator: Optional[Agent] = None):
        self.agents = agents
        self.topology = topology
        self.coordinator = coordinator
        self._round_robin_idx = 0

    def select_next(self, history: list[Message]) -> list[Agent]:
        """
        Returns list of agents who should speak next.

        Round-robin: exactly one agent.
        Coordinator: the coordinator picks one (or itself speaks).
        Broadcast: all agents except the last speaker.
        """
        if self.topology == Topology.ROUND_ROBIN:
            agent = self.agents[self._round_robin_idx % len(self.agents)]
            self._round_robin_idx += 1
            return [agent]

        elif self.topology == Topology.COORDINATOR:
            if not self.coordinator:
                raise ValueError("Coordinator topology requires a coordinator agent")

            # Coordinator speaks first to direct the conversation
            if not history or history[-1].sender != self.coordinator.name:
                return [self.coordinator]

            # Coordinator's last message should name the next speaker
            last_content = history[-1].content.lower()
            for agent in self.agents:
                if agent.name.lower() in last_content and agent != self.coordinator:
                    return [agent]

            # Fallback: coordinator speaks again to redirect
            return [self.coordinator]

        elif self.topology == Topology.BROADCAST:
            last_sender = history[-1].sender if history else None
            return [a for a in self.agents if a.name != last_sender]

        raise ValueError(f"Unknown topology: {self.topology}")


# =============================================================================
# Termination Conditions
# =============================================================================
# Without termination conditions, agents loop forever. These are the "circuit
# breakers" of multi-agent systems.

class TerminationCondition(Protocol):
    def should_stop(self, history: list[Message]) -> bool: ...
    def reason(self) -> str: ...


class MaxRoundsTermination:
    """Stop after N total messages. Simple, reliable, prevents runaway costs."""

    def __init__(self, max_rounds: int):
        self.max_rounds = max_rounds

    def should_stop(self, history: list[Message]) -> bool:
        return len(history) >= self.max_rounds

    def reason(self) -> str:
        return f"Maximum rounds ({self.max_rounds}) reached"


class ConsensusTermination:
    """
    Stop when agents express agreement. Detects consensus through keyword
    matching. In production, you'd use semantic similarity or explicit voting.
    """

    AGREEMENT_PHRASES = [
        "i agree", "consensus reached", "looks good", "approved",
        "let's go with", "that works", "sounds right", "finalize"
    ]

    def __init__(self, required_agreements: int = 2):
        self.required_agreements = required_agreements
        self._triggered = False

    def should_stop(self, history: list[Message]) -> bool:
        if len(history) < 3:
            return False

        # Count agreement signals in the last N messages
        recent = history[-5:]
        agreements = 0
        for msg in recent:
            content_lower = msg.content.lower()
            if any(phrase in content_lower for phrase in self.AGREEMENT_PHRASES):
                agreements += 1

        self._triggered = agreements >= self.required_agreements
        return self._triggered

    def reason(self) -> str:
        return f"Consensus detected ({self.required_agreements} agreements)"


class KeywordTermination:
    """Stop when a specific keyword appears (e.g., 'FINAL ANSWER')."""

    def __init__(self, keyword: str):
        self.keyword = keyword.lower()

    def should_stop(self, history: list[Message]) -> bool:
        if not history:
            return False
        return self.keyword in history[-1].content.lower()

    def reason(self) -> str:
        return f"Keyword '{self.keyword}' detected"


# =============================================================================
# Conversation Manager
# =============================================================================
# The runtime that drives the multi-agent loop. This is the "event loop" of
# the system — it selects agents, collects responses, checks termination,
# and maintains the shared message history.

class ConversationManager:
    """
    Orchestrates a multi-agent conversation.

    The manager is topology-agnostic — it delegates agent selection to the
    AgentSelector and termination checking to TerminationConditions. This
    separation of concerns makes it easy to experiment with different
    configurations without changing the core loop.
    """

    def __init__(
        self,
        agents: list[Agent],
        topology: Topology = Topology.ROUND_ROBIN,
        coordinator: Optional[Agent] = None,
        termination_conditions: list | None = None,
        verbose: bool = True,
    ):
        self.agents = agents
        self.selector = AgentSelector(agents, topology, coordinator)
        self.termination_conditions = termination_conditions or [MaxRoundsTermination(20)]
        self.history: list[Message] = []
        self.verbose = verbose

    def add_message(self, sender: str, content: str, **metadata) -> Message:
        """Add a message to the shared history."""
        msg = Message(sender=sender, content=content, metadata=metadata)
        self.history.append(msg)
        if self.verbose:
            print(f"  {msg}")
        return msg

    def _check_termination(self) -> Optional[str]:
        """Check all termination conditions. Returns reason if should stop."""
        for condition in self.termination_conditions:
            if condition.should_stop(self.history):
                return condition.reason()
        return None

    def run(self, initial_message: str, sender: str = "user") -> list[Message]:
        """
        Run the multi-agent conversation.

        1. Seed with the initial message
        2. Loop: select agent(s) → generate response(s) → check termination
        3. Return full history

        This is the core dispatch loop. In production, step 2 would be async
        to parallelize independent agent calls.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  MULTI-AGENT CONVERSATION")
            print(f"  Topology: {self.selector.topology.value}")
            print(f"  Agents: {', '.join(a.name for a in self.agents)}")
            print(f"{'='*60}\n")

        # Seed the conversation
        self.add_message(sender, initial_message)

        round_num = 0
        while True:
            round_num += 1

            # Check termination BEFORE selecting next agents
            stop_reason = self._check_termination()
            if stop_reason:
                if self.verbose:
                    print(f"\n  [SYSTEM] Conversation ended: {stop_reason}")
                self.add_message("system", f"Conversation ended: {stop_reason}")
                break

            # Select who speaks next
            next_agents = self.selector.select_next(self.history)

            if self.verbose:
                print(f"\n  --- Round {round_num} ---")

            # Collect responses
            any_responded = False
            for agent in next_agents:
                response = agent.respond(self.history)
                if response:
                    self.add_message(agent.name, response)
                    any_responded = True

            # Safety valve: if no agent has anything to say, stop
            if not any_responded:
                if self.verbose:
                    print("  [SYSTEM] No agent responded. Ending conversation.")
                self.add_message("system", "No agent responded. Conversation ended.")
                break

        return self.history

    def get_summary(self) -> str:
        """Generate a summary of the conversation."""
        agent_msg_counts = {}
        for msg in self.history:
            if msg.sender not in ("user", "system"):
                agent_msg_counts[msg.sender] = agent_msg_counts.get(msg.sender, 0) + 1

        lines = [
            f"Conversation Summary:",
            f"  Total messages: {len(self.history)}",
            f"  Agent contributions:",
        ]
        for agent, count in sorted(agent_msg_counts.items()):
            lines.append(f"    {agent}: {count} messages")

        return "\n".join(lines)


# =============================================================================
# Practical Example: Code Review Pipeline
# =============================================================================
# Three agents collaborate to review and improve a code solution:
# - Developer: writes/defends code
# - Reviewer: finds bugs and style issues
# - Architect: makes final decisions on design tradeoffs

def create_code_review_agents() -> tuple[Agent, Agent, Agent]:
    """
    Create a team of agents for code review.

    Each agent has specific behaviors triggered by keywords in the conversation.
    This simulates how LLM-based agents would respond to different contexts.
    """
    developer = Agent(
        name="Developer",
        role="Senior Developer who writes and defends code",
        behaviors={
            "review": "I've written a function to sort user records by multiple fields. "
                      "It uses a stable sort with key functions for each field. "
                      "Time complexity is O(n*k*log(n)) where k is the number of sort fields.",
            "bug": "Good catch. I'll fix the edge case where the sort field is None. "
                   "We should use a default value to handle missing fields: "
                   "key=lambda x: (x.get(field, '') or '') for each field.",
            "performance": "For the performance concern — the current approach creates "
                          "intermediate lists for each sort pass. We could use a single "
                          "composite key function instead: key=lambda x: tuple(x.get(f, '') for f in fields). "
                          "This reduces it to one sort pass: O(n*log(n)).",
            "approve": "I agree with the final approach. The composite key is cleaner and faster. "
                       "I'll update the implementation. Looks good to finalize.",
        }
    )

    reviewer = Agent(
        name="Reviewer",
        role="Code Reviewer focused on correctness, edge cases, and style",
        behaviors={
            "sort": "I see a potential bug: what happens when a sort field has None values? "
                    "Python 3 can't compare None with strings/numbers. This will raise "
                    "TypeError in production. We need null handling.",
            "fix": "The None handling looks correct now. But I have a performance concern: "
                   "running multiple stable sorts is O(k * n*log(n)). For large datasets, "
                   "could we use a composite key instead?",
            "composite": "The composite key approach is much better. One sort pass, cleaner code. "
                        "I approve this approach. Let's finalize.",
            "agree": "I agree. Consensus reached — let's go with the composite key approach.",
        }
    )

    architect = Agent(
        name="Architect",
        role="System Architect who makes design decisions and resolves disagreements",
        behaviors={
            "performance": "From an architecture perspective, the composite key is the right call. "
                          "It's O(n*log(n)) vs O(k*n*log(n)), and it's more readable. "
                          "Developer, please update. Reviewer, any other concerns?",
            "approve": "Approved. The composite key approach is clean, performant, and handles "
                       "nulls correctly. Let's finalize this. FINAL ANSWER: use composite key sort.",
            "consensus": "Good discussion. FINAL ANSWER: use composite key sort with null handling.",
            "review": "Let me see the proposed approach. Developer, walk us through the implementation. "
                      "Reviewer, prepare to review for edge cases.",
        }
    )

    return developer, reviewer, architect


def run_code_review_demo():
    """
    Demonstrate a multi-agent code review using round-robin topology.

    The conversation flows naturally: Developer presents code → Reviewer
    finds issues → Developer fixes → Architect approves.
    """
    developer, reviewer, architect = create_code_review_agents()

    manager = ConversationManager(
        agents=[developer, reviewer, architect],
        topology=Topology.ROUND_ROBIN,
        termination_conditions=[
            MaxRoundsTermination(15),
            KeywordTermination("FINAL ANSWER"),
            ConsensusTermination(required_agreements=2),
        ],
        verbose=True,
    )

    manager.run(
        "Please review this multi-field sort function for our user records system."
    )

    print(f"\n{manager.get_summary()}")
    return manager


# =============================================================================
# Example 2: Coordinator-Based Research Discussion
# =============================================================================

def create_research_agents() -> tuple[Agent, list[Agent]]:
    """
    Create a coordinator and specialist agents for a research discussion.
    The coordinator directs the conversation to the right specialist.
    """
    coordinator = Agent(
        name="Moderator",
        role="Discussion moderator who directs questions to the right expert",
        behaviors={
            "analyze": "Good question. Let me direct this to our specialists. "
                       "Researcher, what does the current literature say about this approach?",
            "literature": "Thank you Researcher. Skeptic, what are the potential flaws "
                         "in this reasoning?",
            "flaws": "Both perspectives are valuable. Researcher, can you address "
                     "the Skeptic's concerns with evidence?",
            "evidence": "I think we've reached a solid conclusion. Both sides have "
                       "been heard. Let's finalize. I agree with this analysis.",
            "perspective": "Good synthesis. I agree we can finalize this discussion. "
                          "FINAL ANSWER: balanced analysis complete.",
        }
    )

    researcher = Agent(
        name="Researcher",
        role="Domain expert who cites evidence and literature",
        behaviors={
            "literature": "The current literature suggests three main approaches. "
                         "Smith et al. (2024) showed a 15% improvement with method A, "
                         "while Jones (2025) found method B scales better beyond 1M records. "
                         "The tradeoff is complexity vs. scalability.",
            "concerns": "To address the Skeptic's concerns: the reproducibility issue "
                       "was resolved in Smith's follow-up paper. The sample size of 10K "
                       "is standard for this type of study. However, I agree the "
                       "generalizability to non-English data is unproven.",
            "address": "The evidence suggests method A for smaller datasets and method B "
                      "for production scale. I agree with this balanced perspective.",
        }
    )

    skeptic = Agent(
        name="Skeptic",
        role="Critical thinker who challenges assumptions and finds flaws",
        behaviors={
            "approach": "I have concerns. First, the sample sizes in those studies "
                       "are relatively small. Second, none of them tested with real "
                       "production data. Third, the improvement margins may not "
                       "justify the added complexity. Let's not rush to conclusions.",
            "improvement": "While the reproducibility concern is addressed, I still "
                          "think the flaws in generalizability are significant. "
                          "We should note this limitation clearly in any recommendation.",
            "resolved": "Fair points from the Researcher. I can agree the evidence "
                       "supports a conditional recommendation. I agree we can finalize.",
        }
    )

    return coordinator, [coordinator, researcher, skeptic]


def run_research_demo():
    """
    Demonstrate coordinator-based topology where the Moderator directs
    the conversation between a Researcher and Skeptic.
    """
    coordinator, agents = create_research_agents()

    manager = ConversationManager(
        agents=agents,
        topology=Topology.COORDINATOR,
        coordinator=coordinator,
        termination_conditions=[
            MaxRoundsTermination(15),
            KeywordTermination("FINAL ANSWER"),
        ],
        verbose=True,
    )

    manager.run(
        "Analyze whether transformer-based models are the best approach for "
        "time-series forecasting in financial markets."
    )

    print(f"\n{manager.get_summary()}")
    return manager


# =============================================================================
# Example 3: Broadcast Topology — Brainstorming
# =============================================================================

def run_brainstorm_demo():
    """
    Demonstrate broadcast topology where all agents respond to each message.
    Good for brainstorming where you want maximum diversity of ideas.
    """
    agents = [
        Agent(
            name="Optimist",
            role="Sees opportunities and upsides",
            behaviors={
                "idea": "This could be huge! Imagine the possibilities for personalized "
                        "education. Students learn at their own pace with AI tutors. "
                        "The market is massive — $5T globally in education.",
                "concern": "While those risks are real, they're solvable. Content moderation "
                          "is a known problem with existing solutions. I approve moving forward.",
                "regulation": "Regulation is coming regardless. Being early means we shape "
                             "the standards. I agree we should proceed with guardrails.",
            }
        ),
        Agent(
            name="Pessimist",
            role="Identifies risks and failure modes",
            behaviors={
                "idea": "Hold on. AI tutors raise serious concerns: misinformation, "
                        "over-reliance on AI, data privacy for minors. Plus, edtech "
                        "has a graveyard of failed startups. What's different here?",
                "opportunities": "The market size doesn't matter if you can't solve the "
                                "trust problem. Parents won't hand their kids to an AI. "
                                "We need heavy regulation compliance and content safety.",
                "guardrails": "If we have proper guardrails, I can cautiously agree. "
                             "But we need human oversight at every step. I agree with "
                             "a guarded approach.",
            }
        ),
        Agent(
            name="Pragmatist",
            role="Focuses on what's actionable and practical",
            behaviors={
                "idea": "Let's scope this down. Instead of 'AI tutor for everything,' "
                        "start with one subject — math — and one grade level. Build a "
                        "prototype, test with 100 students, measure outcomes.",
                "concerns": "Both valid points. Here's the practical path: build with "
                           "guardrails from day one, start small, and iterate. No need "
                           "to solve everything upfront. I agree on a phased approach.",
                "approach": "Agreed. Phase 1: math tutor prototype. Phase 2: expand if "
                           "metrics look good. Phase 3: platform play. Let's finalize. "
                           "I agree with this phased plan.",
            }
        ),
    ]

    manager = ConversationManager(
        agents=agents,
        topology=Topology.BROADCAST,
        termination_conditions=[
            MaxRoundsTermination(12),
            ConsensusTermination(required_agreements=3),
        ],
        verbose=True,
    )

    manager.run(
        "New idea: an AI-powered tutoring platform. Should we pursue this?"
    )

    print(f"\n{manager.get_summary()}")
    return manager


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DEMO 1: Code Review (Round-Robin Topology)")
    print("=" * 60)
    mgr1 = run_code_review_demo()

    print("\n\n")
    print("=" * 60)
    print("  DEMO 2: Research Discussion (Coordinator Topology)")
    print("=" * 60)
    mgr2 = run_research_demo()

    print("\n\n")
    print("=" * 60)
    print("  DEMO 3: Brainstorming (Broadcast Topology)")
    print("=" * 60)
    mgr3 = run_brainstorm_demo()

    # Final summary
    print("\n\n")
    print("=" * 60)
    print("  TOPOLOGY COMPARISON")
    print("=" * 60)
    print(f"\n  Round-Robin: {len(mgr1.history)} messages — sequential, predictable")
    print(f"  Coordinator: {len(mgr2.history)} messages — directed, efficient")
    print(f"  Broadcast:   {len(mgr3.history)} messages — parallel, high-bandwidth")
    print(f"\n  Key insight: topology choice depends on the task.")
    print(f"  - Structured review → round-robin or coordinator")
    print(f"  - Brainstorming → broadcast")
    print(f"  - Complex research → coordinator with domain experts")
