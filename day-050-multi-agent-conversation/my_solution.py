"""
Day 50: Multi-Agent Conversation System — Your Implementation

Build a system where multiple AI agents with distinct roles collaborate
through structured dialogue. Implement message passing, conversation
topologies, and termination conditions.

Hints:
- Start with the Message dataclass — it's the foundation everything else builds on
- The Agent class needs a respond() method that takes history and returns a string or None
- Think of the ConversationManager as an event loop: select → respond → check stop → repeat
- Topologies are just different agent selection strategies
- Termination conditions prevent infinite loops — always have at least one!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol


# =============================================================================
# Message Protocol
# =============================================================================
# Hint: A message needs a sender, content, and timestamp at minimum.
# The metadata dict is for extensibility (confidence scores, etc.)

@dataclass
class Message:
    """A single message in the conversation."""
    sender: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Agent
# =============================================================================
# Hint: An agent is defined by its name, role, and how it responds.
# The behaviors dict maps trigger keywords to response templates.
# In production, you'd replace the keyword matching with an LLM call.

class Agent:
    """An agent with a defined role that generates responses."""

    def __init__(self, name: str, role: str, behaviors: dict[str, str] | None = None):
        """
        Args:
            name: Agent identifier (e.g., "Reviewer", "Architect")
            role: Role description that guides the agent's responses
            behaviors: Dict mapping trigger keywords to response templates
        """
        self.name = name
        self.role = role
        self.behaviors: dict[str, str] = behaviors or {}
        self._response_count = 0

    def respond(self, history: list[Message]) -> Optional[str]:
        """
        Generate a response given the conversation history.

        Scan recent messages for trigger keywords. Return None if the agent
        has nothing to add (prevents echo loops and unnecessary messages).

        Hint: Check the last few messages for trigger keywords from self.behaviors.
        Don't respond to your own messages — check last_msg.sender != self.name.
        """
        raise NotImplementedError("TODO: implement this")

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r})"


# =============================================================================
# Conversation Topologies
# =============================================================================
# Hint: The topology determines who speaks when.
# - Round-robin: cycle through agents in order (use an index counter)
# - Coordinator: a designated agent picks who speaks next
# - Broadcast: all agents except the last speaker respond

class Topology(Enum):
    ROUND_ROBIN = "round_robin"
    COORDINATOR = "coordinator"
    BROADCAST = "broadcast"


class AgentSelector:
    """Selects the next agent(s) to speak based on topology."""

    def __init__(self, agents: list[Agent], topology: Topology,
                 coordinator: Optional[Agent] = None):
        self.agents = agents
        self.topology = topology
        self.coordinator = coordinator
        self._round_robin_idx = 0

    def select_next(self, history: list[Message]) -> list[Agent]:
        """
        Return list of agents who should speak next.

        Hint:
        - ROUND_ROBIN: return one agent, advance the index
        - COORDINATOR: if last speaker isn't coordinator, return coordinator;
          otherwise parse coordinator's message for agent names
        - BROADCAST: return all agents except the last speaker
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Termination Conditions
# =============================================================================
# Hint: Each condition has should_stop(history) -> bool and reason() -> str.
# Combine multiple conditions — any one triggering stops the conversation.

class MaxRoundsTermination:
    """Stop after N total messages."""

    def __init__(self, max_rounds: int):
        self.max_rounds = max_rounds

    def should_stop(self, history: list[Message]) -> bool:
        raise NotImplementedError("TODO: implement this")

    def reason(self) -> str:
        return f"Maximum rounds ({self.max_rounds}) reached"


class ConsensusTermination:
    """Stop when multiple agents express agreement."""

    AGREEMENT_PHRASES = [
        "i agree", "consensus reached", "looks good", "approved",
        "let's go with", "that works", "sounds right", "finalize"
    ]

    def __init__(self, required_agreements: int = 2):
        self.required_agreements = required_agreements

    def should_stop(self, history: list[Message]) -> bool:
        """
        Hint: Look at the last 5 messages. Count how many contain
        agreement phrases. Return True if count >= required_agreements.
        """
        raise NotImplementedError("TODO: implement this")

    def reason(self) -> str:
        return f"Consensus detected ({self.required_agreements} agreements)"


class KeywordTermination:
    """Stop when a specific keyword appears in the last message."""

    def __init__(self, keyword: str):
        self.keyword = keyword.lower()

    def should_stop(self, history: list[Message]) -> bool:
        raise NotImplementedError("TODO: implement this")

    def reason(self) -> str:
        return f"Keyword '{self.keyword}' detected"


# =============================================================================
# Conversation Manager
# =============================================================================
# Hint: This is the core dispatch loop.
# 1. Add the initial message to history
# 2. Loop: check termination → select agents → collect responses → repeat
# 3. If no agent responds, stop (safety valve)

class ConversationManager:
    """Orchestrates a multi-agent conversation."""

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
        raise NotImplementedError("TODO: implement this")

    def run(self, initial_message: str, sender: str = "user") -> list[Message]:
        """
        Run the multi-agent conversation.

        Hint: The loop structure is:
            add initial message
            while True:
                check termination → break if triggered
                select next agent(s)
                for each agent: get response, add to history
                if no agent responded → break
            return history
        """
        raise NotImplementedError("TODO: implement this")

    def get_summary(self) -> str:
        """Generate a summary of the conversation."""
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    # Create a simple round-robin conversation
    agents = [
        Agent(
            name="Developer",
            role="Senior Developer",
            behaviors={
                "review": "Here's my implementation using a composite key sort.",
                "bug": "Good catch, I'll fix the None handling.",
                "approve": "I agree, let's finalize this approach.",
            }
        ),
        Agent(
            name="Reviewer",
            role="Code Reviewer",
            behaviors={
                "sort": "I found a potential bug with None values.",
                "fix": "The fix looks good. I approve this approach.",
                "agree": "I agree. Consensus reached.",
            }
        ),
    ]

    manager = ConversationManager(
        agents=agents,
        topology=Topology.ROUND_ROBIN,
        termination_conditions=[
            MaxRoundsTermination(10),
            ConsensusTermination(required_agreements=2),
        ],
        verbose=True,
    )

    history = manager.run("Please review the sort function for edge cases.")
    print(f"\nTotal messages: {len(history)}")
    print(manager.get_summary())
