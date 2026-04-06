"""
Day 007: State Machine for Robot Behavior

A complete finite state machine implementation with:
1. Generic StateMachine class with states, transitions, guards, entry/exit actions
2. Patrol robot behavior FSM as a concrete example
3. Event-driven transitions with guard conditions
4. State history logging with timestamps
5. Simulation of a realistic patrol scenario

No external dependencies -- just Python stdlib.
"""

from __future__ import annotations

import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from dataclasses import dataclass, field


# =============================================================================
# State Machine Engine
# =============================================================================

@dataclass
class Transition:
    """
    A single transition: from_state --[event, guard?]--> to_state.

    guard: optional callable that returns True if the transition is allowed.
           Receives the FSM context dict as an argument.
    action: optional callable executed when the transition fires.
            Receives the FSM context dict as an argument.
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

    on_enter: called when entering this state (receives context)
    on_exit: called when leaving this state (receives context)
    """
    name: str
    on_enter: Optional[Callable[[Dict[str, Any]], None]] = None
    on_exit: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class TransitionRecord:
    """A record of a single state transition for the history log."""
    timestamp: float
    from_state: str
    event: str
    to_state: str
    guard_passed: bool


class StateMachine:
    """
    Generic event-driven finite state machine.

    Features:
    - Named states with optional entry/exit actions
    - Event-driven transitions with optional guard conditions
    - Transition actions (fire during the transition)
    - Full transition history logging
    - Context dict for sharing data between guards and actions

    Usage:
        sm = StateMachine(initial_state="IDLE")
        sm.add_state(StateConfig("IDLE", on_enter=...))
        sm.add_state(StateConfig("PATROL", on_enter=...))
        sm.add_transition(Transition("IDLE", "start", "PATROL", guard=...))
        sm.handle_event("start")
    """

    def __init__(self, initial_state: str, context: Optional[Dict[str, Any]] = None) -> None:
        self._current_state: str = initial_state
        self._states: Dict[str, StateConfig] = {}
        self._transitions: List[Transition] = []
        self._history: List[TransitionRecord] = []
        self.context: Dict[str, Any] = context if context is not None else {}
        self._start_time: float = time.monotonic()

    @property
    def current_state(self) -> str:
        """Return the current state name."""
        return self._current_state

    @property
    def history(self) -> List[TransitionRecord]:
        """Return the full transition history."""
        return list(self._history)

    @property
    def states(self) -> Set[str]:
        """Return the set of registered state names."""
        return set(self._states.keys())

    def add_state(self, config: StateConfig) -> None:
        """Register a state with optional entry/exit actions."""
        self._states[config.name] = config

    def add_transition(self, transition: Transition) -> None:
        """
        Register a transition.

        Multiple transitions can exist for the same (from_state, event) pair --
        the first one whose guard passes will fire.
        """
        self._transitions.append(transition)

    def get_valid_events(self) -> List[str]:
        """Return list of events that have transitions from the current state."""
        events = []
        for t in self._transitions:
            if t.from_state == self._current_state and t.event not in events:
                events.append(t.event)
        return events

    def handle_event(self, event: str) -> bool:
        """
        Process an event. Returns True if a transition fired, False otherwise.

        Steps:
        1. Find all transitions matching (current_state, event)
        2. For each, evaluate the guard condition (if any)
        3. Fire the first transition whose guard passes
        4. Execute: exit action, transition action, state change, entry action
        5. Log the transition to history
        """
        matching = [
            t for t in self._transitions
            if t.from_state == self._current_state and t.event == event
        ]

        if not matching:
            # No transition defined for this event in current state
            return False

        for transition in matching:
            # Check guard
            guard_passed = True
            if transition.guard is not None:
                guard_passed = transition.guard(self.context)

            if guard_passed:
                old_state = self._current_state
                new_state = transition.to_state
                elapsed = time.monotonic() - self._start_time

                # Exit action for old state
                if old_state in self._states and self._states[old_state].on_exit:
                    self._states[old_state].on_exit(self.context)

                # Transition action
                if transition.action is not None:
                    transition.action(self.context)

                # State change
                self._current_state = new_state

                # Entry action for new state
                if new_state in self._states and self._states[new_state].on_enter:
                    self._states[new_state].on_enter(self.context)

                # Log
                self._history.append(TransitionRecord(
                    timestamp=elapsed,
                    from_state=old_state,
                    event=event,
                    to_state=new_state,
                    guard_passed=True,
                ))

                return True
            else:
                # Log the blocked transition
                elapsed = time.monotonic() - self._start_time
                self._history.append(TransitionRecord(
                    timestamp=elapsed,
                    from_state=self._current_state,
                    event=event,
                    to_state=transition.to_state,
                    guard_passed=False,
                ))

        return False

    def reset(self, initial_state: str) -> None:
        """Reset the machine to a given state, clearing history."""
        self._current_state = initial_state
        self._history.clear()
        self._start_time = time.monotonic()


# =============================================================================
# Patrol Robot FSM
# =============================================================================

def create_patrol_robot() -> StateMachine:
    """
    Create a patrol robot FSM with the following states and transitions:

    IDLE ──start──> PATROL
    PATROL ──obstacle_detected──> OBSTACLE_DETECTED
    PATROL ──battery_low──> RETURN_HOME  (guard: battery < 20)
    OBSTACLE_DETECTED ──avoid──> AVOIDING
    AVOIDING ──clear──> PATROL
    AVOIDING ──battery_low──> RETURN_HOME  (guard: battery < 20)
    RETURN_HOME ──arrived_home──> CHARGING
    CHARGING ──charged──> IDLE  (guard: battery >= 95)
    """
    context = {
        "battery": 100.0,
        "obstacles_avoided": 0,
        "patrol_cycles": 0,
        "log": [],
    }

    sm = StateMachine(initial_state="IDLE", context=context)

    # --- Define states with entry/exit actions ---
    sm.add_state(StateConfig(
        "IDLE",
        on_enter=lambda ctx: ctx["log"].append("Robot is idle, awaiting commands."),
    ))
    sm.add_state(StateConfig(
        "PATROL",
        on_enter=lambda ctx: (
            ctx["log"].append("Patrol started. Scanning environment."),
            ctx.__setitem__("patrol_cycles", ctx["patrol_cycles"] + 1),
        ),
        on_exit=lambda ctx: ctx["log"].append("Patrol paused."),
    ))
    sm.add_state(StateConfig(
        "OBSTACLE_DETECTED",
        on_enter=lambda ctx: ctx["log"].append("Obstacle detected! Analyzing..."),
    ))
    sm.add_state(StateConfig(
        "AVOIDING",
        on_enter=lambda ctx: ctx["log"].append("Executing avoidance maneuver."),
        on_exit=lambda ctx: (
            ctx.__setitem__("obstacles_avoided", ctx["obstacles_avoided"] + 1),
            ctx["log"].append(f"Obstacle avoided (total: {ctx['obstacles_avoided']})."),
        ),
    ))
    sm.add_state(StateConfig(
        "RETURN_HOME",
        on_enter=lambda ctx: ctx["log"].append(f"Returning home. Battery: {ctx['battery']:.0f}%"),
    ))
    sm.add_state(StateConfig(
        "CHARGING",
        on_enter=lambda ctx: ctx["log"].append("Docked. Charging started."),
        on_exit=lambda ctx: ctx["log"].append(f"Charging complete. Battery: {ctx['battery']:.0f}%"),
    ))

    # --- Define transitions ---
    sm.add_transition(Transition("IDLE", "start", "PATROL"))

    sm.add_transition(Transition("PATROL", "obstacle_detected", "OBSTACLE_DETECTED"))

    sm.add_transition(Transition(
        "PATROL", "battery_low", "RETURN_HOME",
        guard=lambda ctx: ctx["battery"] < 20,
    ))

    sm.add_transition(Transition("OBSTACLE_DETECTED", "avoid", "AVOIDING"))

    sm.add_transition(Transition("AVOIDING", "clear", "PATROL"))

    sm.add_transition(Transition(
        "AVOIDING", "battery_low", "RETURN_HOME",
        guard=lambda ctx: ctx["battery"] < 20,
    ))

    sm.add_transition(Transition("RETURN_HOME", "arrived_home", "CHARGING"))

    sm.add_transition(Transition(
        "CHARGING", "charged", "IDLE",
        guard=lambda ctx: ctx["battery"] >= 95,
    ))

    return sm


# =============================================================================
# Simulation
# =============================================================================

def simulate_patrol(verbose: bool = True) -> StateMachine:
    """
    Simulate a complete patrol scenario.

    The robot starts idle, patrols, encounters obstacles, runs low on battery,
    returns home to charge, and goes back to idle.
    """
    robot = create_patrol_robot()

    # Scenario: sequence of (event, battery_delta) pairs
    # battery_delta simulates drain/charge between events
    scenario = [
        ("start",              -5),    # start patrol, drain a bit
        ("obstacle_detected",  -3),    # encounter obstacle
        ("avoid",              -2),    # start avoiding
        ("clear",              -5),    # back to patrol
        ("obstacle_detected",  -3),    # another obstacle
        ("avoid",              -2),    # avoiding
        ("clear",              -10),   # back to patrol, big drain
        ("obstacle_detected",  -5),    # another obstacle
        ("avoid",              -2),    # avoiding
        ("clear",              -20),   # patrol with heavy drain
        ("battery_low",        -5),    # battery getting low (guard will check)
        ("obstacle_detected",  -3),    # one more obstacle
        ("avoid",              -2),    # avoiding
        ("battery_low",        -2),    # try battery_low while avoiding
        ("arrived_home",        0),    # arrive at charging station
        ("charged",            50),    # partially charge (guard: need >= 95)
        ("charged",            40),    # charge more (now battery should be >= 95)
        ("start",              -5),    # patrol again
    ]

    if verbose:
        print(f"\n  Initial state: {robot.current_state}")
        print(f"  Battery: {robot.context['battery']:.0f}%\n")

    for event, battery_delta in scenario:
        # Apply battery change
        robot.context["battery"] = max(0, min(100, robot.context["battery"] + battery_delta))

        old_state = robot.current_state
        result = robot.handle_event(event)

        if verbose:
            status = "OK" if result else "BLOCKED"
            print(
                f"  Event: {event:<22s} | {old_state:<20s} -> "
                f"{robot.current_state:<20s} | Battery: {robot.context['battery']:5.1f}% | {status}"
            )

    return robot


def print_transition_history(sm: StateMachine) -> None:
    """Print the full transition history in a table."""
    print(f"\n  {'#':<4} {'Time':>8} {'From':<22} {'Event':<22} {'To':<22} {'Guard':>6}")
    print("  " + "-" * 86)

    for i, record in enumerate(sm.history):
        guard_str = "pass" if record.guard_passed else "BLOCK"
        print(
            f"  {i+1:<4} {record.timestamp:>7.3f}s "
            f"{record.from_state:<22} {record.event:<22} "
            f"{record.to_state:<22} {guard_str:>6}"
        )


def print_state_timeline(sm: StateMachine, width: int = 60) -> None:
    """Print an ASCII timeline of state occupancy."""
    if not sm.history:
        print("  No transitions recorded.")
        return

    # Collect state intervals
    intervals = []
    # Initial state from time 0
    first_record = sm.history[0]
    if first_record.guard_passed:
        intervals.append((0.0, first_record.timestamp, first_record.from_state))

    for i, record in enumerate(sm.history):
        if not record.guard_passed:
            continue
        start = record.timestamp
        if i + 1 < len(sm.history):
            # Find next successful transition
            end = None
            for j in range(i + 1, len(sm.history)):
                if sm.history[j].guard_passed:
                    end = sm.history[j].timestamp
                    break
            if end is None:
                end = start + 0.1  # small default
        else:
            end = start + 0.1
        intervals.append((start, end, record.to_state))

    if not intervals:
        return

    total_time = max(end for _, end, _ in intervals)
    if total_time == 0:
        total_time = 1.0

    # Collect unique states in order
    seen = []
    for _, _, state in intervals:
        if state not in seen:
            seen.append(state)
    # Add initial state if not there
    initial = sm.history[0].from_state if sm.history[0].guard_passed else sm.current_state
    if initial not in seen:
        seen.insert(0, initial)

    state_symbols = {}
    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, s in enumerate(seen):
        state_symbols[s] = symbols[i % len(symbols)]

    print("\n  State Timeline:")
    print(f"  {'State':<22} Symbol")
    print("  " + "-" * 32)
    for state in seen:
        print(f"  {state:<22} [{state_symbols[state]}]")

    # Build timeline
    timeline = [" "] * width
    for start, end, state in intervals:
        col_start = int(start / total_time * (width - 1))
        col_end = int(end / total_time * (width - 1))
        col_end = max(col_end, col_start + 1)
        for c in range(col_start, min(col_end, width)):
            timeline[c] = state_symbols[state]

    print(f"\n  |{''.join(timeline)}|")
    print(f"  0{'':>{width - 1}}end")


# =============================================================================
# Main: Demonstrate Everything
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 007: STATE MACHINE FOR ROBOT BEHAVIOR")
    print("=" * 70)

    # --- Basic StateMachine demo ---
    print("\n" + "=" * 70)
    print("PART 1: GENERIC STATE MACHINE DEMO")
    print("=" * 70)

    sm = StateMachine(initial_state="OFF")
    sm.add_state(StateConfig("OFF"))
    sm.add_state(StateConfig("ON"))
    sm.add_transition(Transition("OFF", "press_button", "ON"))
    sm.add_transition(Transition("ON", "press_button", "OFF"))

    print(f"\n  Simple toggle: {sm.current_state}", end="")
    sm.handle_event("press_button")
    print(f" -> {sm.current_state}", end="")
    sm.handle_event("press_button")
    print(f" -> {sm.current_state}")

    # --- Guard condition demo ---
    print("\n" + "=" * 70)
    print("PART 2: GUARD CONDITIONS")
    print("=" * 70)

    sm2 = StateMachine(initial_state="LOCKED", context={"has_key": False})
    sm2.add_state(StateConfig("LOCKED"))
    sm2.add_state(StateConfig("UNLOCKED"))
    sm2.add_transition(Transition(
        "LOCKED", "try_open", "UNLOCKED",
        guard=lambda ctx: ctx["has_key"],
    ))

    print(f"\n  State: {sm2.current_state}, has_key: {sm2.context['has_key']}")
    result = sm2.handle_event("try_open")
    print(f"  try_open -> transition fired: {result}, state: {sm2.current_state}")

    sm2.context["has_key"] = True
    print(f"\n  State: {sm2.current_state}, has_key: {sm2.context['has_key']}")
    result = sm2.handle_event("try_open")
    print(f"  try_open -> transition fired: {result}, state: {sm2.current_state}")

    # --- Invalid event demo ---
    print("\n" + "=" * 70)
    print("PART 3: INVALID EVENTS")
    print("=" * 70)

    sm3 = StateMachine(initial_state="A")
    sm3.add_state(StateConfig("A"))
    sm3.add_state(StateConfig("B"))
    sm3.add_transition(Transition("A", "go", "B"))

    print(f"\n  State: {sm3.current_state}")
    result = sm3.handle_event("fly")  # no transition defined
    print(f"  Event 'fly' -> transition fired: {result}, state: {sm3.current_state}")
    result = sm3.handle_event("go")
    print(f"  Event 'go'  -> transition fired: {result}, state: {sm3.current_state}")
    result = sm3.handle_event("go")  # no transition from B on 'go'
    print(f"  Event 'go' (from B) -> transition fired: {result}, state: {sm3.current_state}")

    # --- Full patrol robot simulation ---
    print("\n" + "=" * 70)
    print("PART 4: PATROL ROBOT SIMULATION")
    print("=" * 70)

    robot = simulate_patrol(verbose=True)

    print(f"\n  Final state: {robot.current_state}")
    print(f"  Battery: {robot.context['battery']:.0f}%")
    print(f"  Obstacles avoided: {robot.context['obstacles_avoided']}")
    print(f"  Patrol cycles: {robot.context['patrol_cycles']}")

    # --- Transition history ---
    print("\n" + "=" * 70)
    print("PART 5: TRANSITION HISTORY")
    print("=" * 70)

    print_transition_history(robot)

    # --- State timeline ---
    print("\n" + "=" * 70)
    print("PART 6: STATE TIMELINE (ASCII)")
    print("=" * 70)

    print_state_timeline(robot)

    # --- Valid events ---
    print("\n" + "=" * 70)
    print("PART 7: VALID EVENTS FROM CURRENT STATE")
    print("=" * 70)

    print(f"\n  Current state: {robot.current_state}")
    print(f"  Valid events: {robot.get_valid_events()}")

    # --- Action log ---
    print("\n" + "=" * 70)
    print("PART 8: ACTION LOG")
    print("=" * 70)

    print()
    for i, msg in enumerate(robot.context["log"]):
        print(f"  {i+1:2d}. {msg}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("- FSMs decompose complex behavior into discrete, manageable states")
    print("- Guard conditions prevent invalid transitions (e.g., charging when battery is full)")
    print("- Entry/exit actions keep state-specific logic contained")
    print("- Transition history is essential for debugging real robot behavior")
    print("- The generic StateMachine class is reusable for any domain")
