"""
Day 007: State Machine for Robot Behavior — Your Implementation

Implement a generic FSM engine and a patrol robot behavior controller.

Key concepts:
    - States: discrete modes the system can be in
    - Events: triggers that cause transitions
    - Guards: boolean conditions that must be True for a transition to fire
    - Actions: code that runs on state entry, exit, or during transitions
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Transition:
    """
    A single transition: from_state --[event, guard?]--> to_state.

    guard: optional callable(context) -> bool
    action: optional callable(context) -> None, runs when transition fires
    """
    from_state: str
    event: str
    to_state: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class StateConfig:
    """
    Configuration for a single state.

    on_enter: called when entering this state (receives context dict)
    on_exit: called when leaving this state (receives context dict)
    """
    name: str
    on_enter: Optional[Callable[[Dict[str, Any]], None]] = None
    on_exit: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class TransitionRecord:
    """Record of a state transition for history logging."""
    timestamp: float
    from_state: str
    event: str
    to_state: str
    guard_passed: bool


# =============================================================================
# State Machine Engine
# =============================================================================

class StateMachine:
    """
    Generic event-driven finite state machine.

    TODO: Implement:
    - __init__: set initial state, create empty states dict, transitions list, history
    - add_state: register a StateConfig
    - add_transition: register a Transition
    - current_state property: return current state name
    - history property: return list of TransitionRecords
    - states property: return set of registered state names
    - get_valid_events: return events with transitions from current state
    - handle_event: process an event, checking guards, firing actions, logging
    - reset: reset to a given state, clear history

    Hints:
    - Store states in a dict[str, StateConfig]
    - Store transitions in a list (multiple transitions can share from_state + event)
    - handle_event should: find matching transitions, check guards, fire first passing one
    - Order of execution: old_state.on_exit -> transition.action -> state change -> new_state.on_enter
    """

    def __init__(self, initial_state: str, context: Optional[Dict[str, Any]] = None) -> None:
        # TODO: Initialize:
        # - _current_state (str)
        # - _states (dict)
        # - _transitions (list)
        # - _history (list)
        # - context (dict)
        # - _start_time (for timestamps)
        raise NotImplementedError

    @property
    def current_state(self) -> str:
        """Return the current state name."""
        raise NotImplementedError

    @property
    def history(self) -> List[TransitionRecord]:
        """Return a copy of the transition history."""
        raise NotImplementedError

    @property
    def states(self) -> Set[str]:
        """Return the set of registered state names."""
        raise NotImplementedError

    def add_state(self, config: StateConfig) -> None:
        """Register a state configuration."""
        raise NotImplementedError

    def add_transition(self, transition: Transition) -> None:
        """Register a transition."""
        raise NotImplementedError

    def get_valid_events(self) -> List[str]:
        """Return events that have transitions from the current state."""
        raise NotImplementedError

    def handle_event(self, event: str) -> bool:
        """
        Process an event. Return True if a transition fired, False otherwise.

        Steps:
        1. Find transitions matching (current_state, event)
        2. If none found, return False
        3. For each matching transition, check guard (if any)
        4. Fire the first transition whose guard passes:
           a. Call on_exit of old state (if defined)
           b. Call transition action (if defined)
           c. Change _current_state
           d. Call on_enter of new state (if defined)
        5. Log a TransitionRecord to history
        6. Return True

        If all guards fail, log a blocked record and return False.
        """
        raise NotImplementedError

    def reset(self, initial_state: str) -> None:
        """Reset machine to given state and clear history."""
        raise NotImplementedError


def create_patrol_robot() -> StateMachine:
    """
    Create a patrol robot FSM.

    TODO: Build an FSM with these states and transitions:
        IDLE --start--> PATROL
        PATROL --obstacle_detected--> OBSTACLE_DETECTED
        PATROL --battery_low--> RETURN_HOME  (guard: battery < 20)
        OBSTACLE_DETECTED --avoid--> AVOIDING
        AVOIDING --clear--> PATROL
        AVOIDING --battery_low--> RETURN_HOME  (guard: battery < 20)
        RETURN_HOME --arrived_home--> CHARGING
        CHARGING --charged--> IDLE  (guard: battery >= 95)

    Context should include: battery (float), obstacles_avoided (int),
    patrol_cycles (int), log (list of strings).

    Add entry/exit actions that append descriptive messages to context["log"].
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("Implement StateMachine and create_patrol_robot.")
    print("Then run: python3 tests.py")
