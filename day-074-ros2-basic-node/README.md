# Day 74: ROS2 Basic Node — Pub/Sub Communication System

## Overview

**Robot Operating System 2 (ROS2)** is the backbone of modern robotics software — it's how robots think, communicate, and coordinate. Today we build the core communication primitives from scratch in pure Python: **nodes**, **topics**, **publishers**, **subscribers**, and **services**.

This isn't just an academic exercise. Every production robot — from warehouse AMRs to surgical arms — runs on these exact patterns. Understanding them deeply means you can debug real robot systems, design clean architectures, and reason about timing and data flow.

We'll simulate the ROS2 middleware (DDS — Data Distribution Service) to understand how decoupled, asynchronous communication works in robotics.

## Core Concepts

### 1. Nodes — The Unit of Computation

A **node** is a single process that does one thing well. A robot might have:
- A `camera_node` that publishes images
- A `detector_node` that subscribes to images and publishes detections
- A `controller_node` that subscribes to detections and publishes motor commands

**Why nodes?** Modularity. You can restart the camera driver without killing the controller. You can swap a LIDAR node for a depth camera node without touching anything else. Each node is independently deployable, testable, and debuggable.

In ROS2, nodes communicate through a **middleware layer** (DDS) — they don't call each other directly. This decoupling is the key design insight.

### 2. Topics — Named Communication Channels

A **topic** is a named bus that carries messages of a specific type:
- `/camera/image` → carries `Image` messages
- `/cmd_vel` → carries `Twist` (velocity) messages
- `/scan` → carries `LaserScan` messages

Topics use **publish/subscribe** semantics:
- Publishers push messages onto a topic
- Subscribers receive messages from a topic
- Neither knows about the other — they only know the topic name and message type

This is fundamentally different from function calls. With pub/sub:
- **Many-to-many**: Multiple publishers and subscribers on the same topic
- **Asynchronous**: Publishers don't block waiting for subscribers
- **Decoupled**: Adding a new subscriber requires zero changes to the publisher

### 3. Messages — Typed Data Structures

Messages are strongly-typed data containers. In ROS2, they're defined in `.msg` files:

```
# geometry_msgs/Twist.msg
Vector3 linear    # linear velocity (x, y, z)
Vector3 angular   # angular velocity (x, y, z)
```

**Why typed?** Catches integration bugs at connection time, not runtime. If node A publishes `Image` on `/data` but node B subscribes expecting `LaserScan`, the system rejects the connection immediately.

### 4. Quality of Service (QoS) — Delivery Guarantees

ROS2 inherits DDS's QoS profiles, which control:
- **Reliability**: `RELIABLE` (TCP-like, guaranteed delivery) vs `BEST_EFFORT` (UDP-like, may drop)
- **Durability**: `TRANSIENT_LOCAL` (late subscribers get the last value) vs `VOLATILE` (miss it and it's gone)
- **History depth**: How many messages to buffer

**The tradeoff**: RELIABLE + deep history = safe but slow. BEST_EFFORT + depth 1 = fast but lossy. Sensor data (camera, LIDAR) typically uses BEST_EFFORT because you want the freshest data, not every frame. Commands use RELIABLE because you can't afford to drop a "stop" message.

### 5. Services — Request/Reply Pattern

Not everything fits pub/sub. Sometimes you need synchronous request/reply:
- "Set parameter X to value Y" → "OK, done"
- "Compute path from A to B" → "Here's the path"

Services provide this. A **service server** advertises a named service with request/reply types. A **service client** sends a request and blocks until the reply arrives.

**When to use services vs topics:**
- Topics: continuous data streams (sensors, commands, state)
- Services: infrequent, one-shot operations (config changes, computations)

### 6. Executors — The Event Loop

Nodes don't just sit there — they need to be *spun*. An **executor** is the event loop that:
1. Checks for incoming messages on all subscriptions
2. Calls the appropriate callback functions
3. Handles timers (periodic callbacks)
4. Processes service requests

ROS2 has two executor models:
- **SingleThreadedExecutor**: One callback at a time (simple, safe)
- **MultiThreadedExecutor**: Multiple callbacks concurrently (fast, requires thread safety)

## Step-by-Step Breakdown

### Step 1: Message System
Define a base `Message` class and concrete message types (`String`, `Twist`, `LaserScan`). Messages must be serializable and typed so the system can enforce type safety on topic connections.

### Step 2: Topic Registry
Build a global `TopicRegistry` that tracks all active topics, their message types, publishers, and subscribers. This is the "switchboard" that routes messages. Without it, publishers would need direct references to subscribers, defeating the decoupling purpose.

### Step 3: Publisher/Subscriber
Implement `Publisher` (pushes messages to a topic with QoS) and `Subscription` (receives messages via callback). The QoS compatibility check happens at connection time — mismatched QoS profiles should warn or reject.

### Step 4: Node
Build the `Node` class that owns publishers, subscribers, timers, and service servers/clients. A node is the organizational unit — it has a name, a namespace, and manages its own lifecycle.

### Step 5: Services
Implement `ServiceServer` and `ServiceClient` for request/reply communication. The server registers a callback; the client blocks until the callback returns a response.

### Step 6: Executor
Build a `SingleThreadedExecutor` that spins nodes — polling for messages, firing timer callbacks, and processing service requests in a round-robin loop.

### Step 7: Demo System
Wire up a multi-node robot system: a sensor node publishing LIDAR scans, a controller node subscribing and publishing velocity commands, and a logger node recording everything.

## Learning Objectives

- Understand **publish/subscribe** as a communication pattern and why robotics chose it over RPC
- Implement **typed message passing** with QoS guarantees
- Build a working **executor/event loop** that drives asynchronous node communication
- Learn the **node-topic-message** architecture that underpins all ROS2 systems
- Understand **services** as the complement to pub/sub for synchronous operations
- Reason about **QoS tradeoffs** between reliability and latency

## Going Deeper

- **Lifecycle nodes**: ROS2 managed nodes have states (unconfigured → inactive → active → finalized). This enables coordinated startup/shutdown — critical when one node depends on another being ready.
- **Actions**: Long-running tasks (navigate to waypoint, pick up object) use the Action pattern — goal/feedback/result, with preemption support. Think of it as a service that sends progress updates.
- **Component composition**: In production, multiple nodes run in the same process as "components" to avoid IPC overhead while keeping logical separation.
- **DDS discovery**: Real ROS2 uses DDS multicast discovery — nodes find each other automatically on the network without a central broker (unlike ROS1's `rosmaster`).
- **Real-time constraints**: ROS2's executor can be paired with real-time schedulers for hard real-time guarantees on callback latency — essential for high-speed control loops.
