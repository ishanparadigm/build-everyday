"""
Day 007: State Machine for Robot Behavior — Test Suite

Tests import from my_solution. Run with: python3 tests.py
"""

import sys

from my_solution import (
    StateMachine,
    StateConfig,
    Transition,
    TransitionRecord,
    create_patrol_robot,
)


def test_initial_state():
    """StateMachine should start in the initial state."""
    sm = StateMachine(initial_state="IDLE")
    assert sm.current_state == "IDLE", f"Expected IDLE, got {sm.current_state}"
    print("  PASS: Initial state is correct")


def test_simple_transition():
    """A basic transition should change the state."""
    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A"))
    sm.add_state(StateConfig("B"))
    sm.add_transition(Transition("A", "go", "B"))

    result = sm.handle_event("go")
    assert result is True, "Transition should have fired"
    assert sm.current_state == "B", f"Expected B, got {sm.current_state}"
    print("  PASS: Simple transition works")


def test_invalid_event_rejected():
    """An event with no matching transition should return False and not change state."""
    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A"))
    sm.add_transition(Transition("A", "go", "B"))

    result = sm.handle_event("fly")
    assert result is False, "Invalid event should return False"
    assert sm.current_state == "A", f"State should not change, got {sm.current_state}"
    print("  PASS: Invalid event rejected")


def test_no_transition_from_state():
    """Event valid in another state but not current state should be rejected."""
    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A"))
    sm.add_state(StateConfig("B"))
    sm.add_transition(Transition("B", "go", "A"))  # only from B

    result = sm.handle_event("go")  # we're in A, not B
    assert result is False, "Should not fire from wrong state"
    assert sm.current_state == "A", f"State should not change, got {sm.current_state}"
    print("  PASS: No transition from current state")


def test_guard_blocks_transition():
    """Guard returning False should block the transition."""
    sm = StateMachine(initial_state="LOCKED", context={"has_key": False})
    sm.add_state(StateConfig("LOCKED"))
    sm.add_state(StateConfig("UNLOCKED"))
    sm.add_transition(Transition(
        "LOCKED", "try_open", "UNLOCKED",
        guard=lambda ctx: ctx["has_key"],
    ))

    result = sm.handle_event("try_open")
    assert result is False, "Guard should block transition"
    assert sm.current_state == "LOCKED", f"State should be LOCKED, got {sm.current_state}"
    print("  PASS: Guard blocks transition when condition is False")


def test_guard_allows_transition():
    """Guard returning True should allow the transition."""
    sm = StateMachine(initial_state="LOCKED", context={"has_key": True})
    sm.add_state(StateConfig("LOCKED"))
    sm.add_state(StateConfig("UNLOCKED"))
    sm.add_transition(Transition(
        "LOCKED", "try_open", "UNLOCKED",
        guard=lambda ctx: ctx["has_key"],
    ))

    result = sm.handle_event("try_open")
    assert result is True, "Guard should allow transition"
    assert sm.current_state == "UNLOCKED", f"Expected UNLOCKED, got {sm.current_state}"
    print("  PASS: Guard allows transition when condition is True")


def test_entry_exit_actions():
    """Entry and exit actions should fire during transitions."""
    log = []

    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A", on_exit=lambda ctx: log.append("exit_A")))
    sm.add_state(StateConfig("B", on_enter=lambda ctx: log.append("enter_B")))
    sm.add_transition(Transition("A", "go", "B"))

    sm.handle_event("go")

    assert "exit_A" in log, f"Exit action should have fired, log: {log}"
    assert "enter_B" in log, f"Entry action should have fired, log: {log}"
    assert log.index("exit_A") < log.index("enter_B"), "Exit should fire before enter"
    print("  PASS: Entry and exit actions fire in correct order")


def test_transition_action():
    """Transition action should fire between exit and enter."""
    log = []

    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A", on_exit=lambda ctx: log.append("exit_A")))
    sm.add_state(StateConfig("B", on_enter=lambda ctx: log.append("enter_B")))
    sm.add_transition(Transition("A", "go", "B", action=lambda ctx: log.append("transition")))

    sm.handle_event("go")

    assert log == ["exit_A", "transition", "enter_B"], f"Expected correct order, got {log}"
    print("  PASS: Transition action fires between exit and enter")


def test_history_recorded():
    """Transition history should be recorded."""
    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A"))
    sm.add_state(StateConfig("B"))
    sm.add_state(StateConfig("C"))
    sm.add_transition(Transition("A", "go", "B"))
    sm.add_transition(Transition("B", "go", "C"))

    sm.handle_event("go")
    sm.handle_event("go")

    history = sm.history
    assert len(history) == 2, f"Expected 2 records, got {len(history)}"
    assert history[0].from_state == "A" and history[0].to_state == "B"
    assert history[1].from_state == "B" and history[1].to_state == "C"
    print("  PASS: State history is recorded")


def test_history_records_blocked():
    """Blocked transitions (failed guards) should be logged with guard_passed=False."""
    sm = StateMachine(initial_state="A", context={"allowed": False})
    sm.add_state(StateConfig("A"))
    sm.add_state(StateConfig("B"))
    sm.add_transition(Transition("A", "go", "B", guard=lambda ctx: ctx["allowed"]))

    sm.handle_event("go")

    history = sm.history
    assert len(history) >= 1, "Blocked transition should be in history"
    assert history[0].guard_passed is False, "Blocked record should have guard_passed=False"
    print("  PASS: Blocked transitions logged in history")


def test_get_valid_events():
    """get_valid_events should return events available from current state."""
    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A"))
    sm.add_state(StateConfig("B"))
    sm.add_transition(Transition("A", "go", "B"))
    sm.add_transition(Transition("A", "run", "B"))
    sm.add_transition(Transition("B", "back", "A"))

    events = sm.get_valid_events()
    assert "go" in events, f"'go' should be valid, got {events}"
    assert "run" in events, f"'run' should be valid, got {events}"
    assert "back" not in events, f"'back' should not be valid from A, got {events}"
    print("  PASS: get_valid_events returns correct events")


def test_reset_clears_state():
    """Reset should change current state and clear history."""
    sm = StateMachine(initial_state="A")
    sm.add_state(StateConfig("A"))
    sm.add_state(StateConfig("B"))
    sm.add_transition(Transition("A", "go", "B"))

    sm.handle_event("go")
    assert sm.current_state == "B"
    assert len(sm.history) > 0

    sm.reset("A")
    assert sm.current_state == "A", f"Expected A after reset, got {sm.current_state}"
    assert len(sm.history) == 0, f"History should be empty after reset, got {len(sm.history)}"
    print("  PASS: Reset clears state and history")


def test_patrol_robot_basic_flow():
    """Patrol robot should handle the basic IDLE -> PATROL -> obstacle flow."""
    robot = create_patrol_robot()

    assert robot.current_state == "IDLE"
    robot.handle_event("start")
    assert robot.current_state == "PATROL", f"Expected PATROL, got {robot.current_state}"

    robot.handle_event("obstacle_detected")
    assert robot.current_state == "OBSTACLE_DETECTED"

    robot.handle_event("avoid")
    assert robot.current_state == "AVOIDING"

    robot.handle_event("clear")
    assert robot.current_state == "PATROL"
    print("  PASS: Patrol robot basic flow works")


def test_patrol_robot_battery_guard():
    """Battery guard should block RETURN_HOME when battery is high."""
    robot = create_patrol_robot()
    robot.handle_event("start")

    # Battery is 100%, should NOT go to RETURN_HOME
    result = robot.handle_event("battery_low")
    assert robot.current_state == "PATROL", \
        f"Should stay in PATROL with high battery, got {robot.current_state}"

    # Drain battery
    robot.context["battery"] = 15.0
    result = robot.handle_event("battery_low")
    assert robot.current_state == "RETURN_HOME", \
        f"Should go to RETURN_HOME with low battery, got {robot.current_state}"
    print("  PASS: Battery guard works correctly")


def test_patrol_robot_charging_guard():
    """Charging guard should block IDLE until battery >= 95."""
    robot = create_patrol_robot()
    robot.handle_event("start")
    robot.context["battery"] = 10.0
    robot.handle_event("battery_low")
    robot.handle_event("arrived_home")

    assert robot.current_state == "CHARGING"

    # Try to finish charging with low battery
    robot.context["battery"] = 50.0
    result = robot.handle_event("charged")
    assert robot.current_state == "CHARGING", \
        f"Should stay CHARGING at 50%, got {robot.current_state}"

    # Full charge
    robot.context["battery"] = 100.0
    result = robot.handle_event("charged")
    assert robot.current_state == "IDLE", \
        f"Should go to IDLE at 100%, got {robot.current_state}"
    print("  PASS: Charging guard works correctly")


def test_states_property():
    """states property should return registered state names."""
    sm = StateMachine(initial_state="X")
    sm.add_state(StateConfig("X"))
    sm.add_state(StateConfig("Y"))
    sm.add_state(StateConfig("Z"))

    states = sm.states
    assert states == {"X", "Y", "Z"}, f"Expected {{X, Y, Z}}, got {states}"
    print("  PASS: states property returns registered states")


if __name__ == "__main__":
    tests = [
        test_initial_state,
        test_simple_transition,
        test_invalid_event_rejected,
        test_no_transition_from_state,
        test_guard_blocks_transition,
        test_guard_allows_transition,
        test_entry_exit_actions,
        test_transition_action,
        test_history_recorded,
        test_history_records_blocked,
        test_get_valid_events,
        test_reset_clears_state,
        test_patrol_robot_basic_flow,
        test_patrol_robot_battery_guard,
        test_patrol_robot_charging_guard,
        test_states_property,
    ]

    print(f"\nRunning {len(tests)} tests for Day 007: State Machine\n")

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} -- {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
