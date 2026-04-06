# Day 007: State Machine for Robot Behavior

## What You're Building

A generic finite state machine (FSM) engine and a concrete robot behavior controller built on top of it. You'll implement states, transitions, guard conditions, entry/exit actions, and event-driven logic -- the backbone of how real robots decide what to do next.

State machines are the workhorse of robotics behavior planning. Every robot that patrols, cleans, delivers, or explores uses some form of FSM to manage its high-level behavior. Understanding FSMs deeply means understanding how to make software that interacts with the messy, unpredictable physical world in a structured, debuggable way.

## Core Concepts

### Finite State Machines

A finite state machine is defined by:

```
FSM = (S, E, T, s0, A)
```

Where:
- **S** = finite set of states (e.g., IDLE, PATROL, AVOID_OBSTACLE)
- **E** = finite set of events/inputs (e.g., start_button, obstacle_detected, battery_low)
- **T** = transition function: S x E -> S (given current state and event, what's the next state?)
- **s0** = initial state
- **A** = actions (things that happen on transitions, state entry, or state exit)

The key property: the machine is always in exactly one state. Transitions are atomic -- you're never "between" states. This makes FSMs easy to reason about, test, and debug.

### Why FSMs for Robotics?

Robots face a fundamental challenge: they must act in real-time in an uncertain environment. FSMs solve this by:

1. **Decomposing behavior into manageable pieces**. Each state handles one concern (patrol, avoid, charge). You don't need to think about obstacle avoidance while writing the charging logic.

2. **Making transitions explicit**. Every state change has a clear trigger and (optionally) a guard condition. No hidden mode switches, no "how did we get here?" debugging sessions.

3. **Enabling systematic testing**. You can enumerate all states and transitions, test each one, and verify the machine can't get stuck in an invalid configuration.

4. **Providing a visual language**. State transition diagrams are immediately understandable by non-programmers -- mechanical engineers, product managers, customers.

### State Transition Diagrams

A patrol robot's behavior:

```
                    start
    [IDLE] ────────────────> [PATROL]
      ^                        |   ^
      |                        |   |
      |   charged         obstacle  clear
      |                   detected  |
      |                        |   |
  [CHARGING] <── low     [OBSTACLE_DETECTED]
      ^        battery         |
      |                        |
      |                     avoid
  [RETURN_HOME]                |
      ^                   [AVOIDING]
      |                        |
      +────── low_battery ─────+
```

Each arrow is a transition triggered by an event. The labels on arrows are the events. Guard conditions (not shown) are boolean checks that must be true for the transition to fire.

### Guard Conditions

Guards add conditional logic to transitions:

```
PATROL --[battery_low, guard: battery < 20%]--> RETURN_HOME
```

The transition only fires if both:
1. The event `battery_low` occurs, AND
2. The guard `battery < 20%` evaluates to True

This prevents transitions that don't make sense. You wouldn't return home if battery is at 80% just because a sensor flickered.

### Entry and Exit Actions

States can have actions that fire automatically:

```
State PATROL:
    on_enter: activate_lidar(), start_mapping()
    on_exit:  stop_mapping(), log_patrol_area()
```

Entry actions initialize the state's behavior. Exit actions clean up. This pattern keeps state-specific logic contained -- you don't scatter initialization code across every transition into PATROL.

### Hierarchical State Machines

Real robots often have nested behaviors:

```
PATROL (superstate)
    ├── MOVING_FORWARD
    ├── TURNING
    └── SCANNING
```

The robot is in PATROL, but within PATROL it cycles through sub-states. If an `obstacle_detected` event fires, it exits the entire PATROL superstate regardless of which sub-state is active.

Hierarchical FSMs (also called statecharts) reduce transition explosion -- instead of connecting every sub-state to OBSTACLE_DETECTED, you connect the parent state once.

## Step-by-Step Approach

### Step 1: Implement the generic StateMachine class
Build a state machine engine with states, events, transitions, guards, and actions. It should be reusable for any domain.

### Step 2: Define the patrol robot states and transitions
Create a concrete FSM for a patrol robot with states like IDLE, PATROL, OBSTACLE_DETECTED, AVOIDING, LOW_BATTERY, RETURN_HOME, CHARGING.

### Step 3: Implement event-driven transitions with guards
Events trigger transitions only if the corresponding guard condition passes. Invalid events in the current state should be rejected gracefully.

### Step 4: Add state history and logging
Record every transition: timestamp, from-state, event, to-state. This is essential for debugging real robot behavior.

### Step 5: Simulate a patrol scenario
Run the robot through a realistic sequence of events and display the state history.

## Learning Objectives

- Implement a generic finite state machine with states, events, transitions, and guards
- Understand why FSMs are the standard approach for robot behavior planning
- Build event-driven transitions with guard conditions
- Implement entry/exit actions for states
- Add state history logging for debugging and analysis
- Simulate a robot patrol scenario end-to-end

## Going Deeper

- **Behavior Trees**: The successor to FSMs in game AI and robotics. Better composability but more complex. Used in Unreal Engine and ROS 2 (BehaviorTree.CPP).
- **Hierarchical State Machines (Statecharts)**: Nested states that reduce transition complexity. David Harel's original 1987 paper is worth reading.
- **SMACH**: The ROS state machine library. If you're building real ROS robots, this is the standard tool.
- **Petri Nets**: A more expressive formalism that handles concurrency (multiple active states). Used in manufacturing and logistics.
- **Decision-theoretic planning**: POMDPs and MDPs -- when you need to handle uncertainty probabilistically rather than with discrete states. The "AI" approach to behavior planning.
- **Subsumption architecture**: Rodney Brooks' approach -- layered behaviors that can suppress each other. Historically important for mobile robots.
