"""
Day 74: ROS2 Basic Node — Pub/Sub Communication System

A pure Python implementation of ROS2's core communication primitives:
nodes, topics, publishers, subscribers, services, and an executor.

This simulates how real ROS2 middleware (DDS) works, teaching the
architecture that runs on virtually every modern robot.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum
from collections import deque


# =============================================================================
# Part 1: Message System
# =============================================================================
# Messages are typed data containers. In real ROS2, these are generated from
# .msg files. We define them as dataclasses for clean, typed Python.

@dataclass
class Message:
    """Base class for all ROS2 messages.

    Every message carries a timestamp (header) so subscribers know
    when the data was generated — critical for sensor fusion where
    you need to align data from different sensors in time.
    """
    timestamp: float = field(default_factory=time.time)


@dataclass
class StringMsg(Message):
    """Simple string message — equivalent to std_msgs/String."""
    data: str = ""


@dataclass
class Twist(Message):
    """Velocity command — equivalent to geometry_msgs/Twist.

    The standard way to command robot motion. Linear is translational
    velocity (m/s), angular is rotational velocity (rad/s).
    For a differential-drive robot, you typically only use:
      linear_x (forward/backward) and angular_z (turning).
    """
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class LaserScan(Message):
    """LIDAR scan data — equivalent to sensor_msgs/LaserScan.

    A 2D LIDAR sweeps a laser beam in a plane and measures distances.
    - angle_min/max: the angular range of the scan (radians)
    - angle_increment: angular step between consecutive rays
    - ranges: list of distances for each ray (meters)
    - range_min/max: valid measurement bounds

    Values outside [range_min, range_max] are invalid (e.g., too close
    or too far for the sensor).
    """
    angle_min: float = -3.14159
    angle_max: float = 3.14159
    angle_increment: float = 0.01745  # ~1 degree
    range_min: float = 0.1
    range_max: float = 30.0
    ranges: List[float] = field(default_factory=list)


@dataclass
class Odometry(Message):
    """Robot odometry — position and velocity estimate.

    In real robots, this comes from wheel encoders, IMU, or visual odometry.
    It represents the robot's best estimate of its own position and velocity
    in the world frame.
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # heading in radians
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


# =============================================================================
# Part 2: Quality of Service (QoS)
# =============================================================================
# QoS profiles control HOW messages are delivered. This is one of ROS2's
# biggest improvements over ROS1 — you can tune reliability vs latency
# per-topic based on what the data needs.

class Reliability(Enum):
    """How hard the system tries to deliver each message.

    RELIABLE: Every message is delivered, retransmitting if needed.
              Use for commands, configuration, state changes.
    BEST_EFFORT: Messages may be dropped if the network is busy.
                 Use for high-frequency sensor data where freshness > completeness.
    """
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class Durability(Enum):
    """What happens to late subscribers.

    TRANSIENT_LOCAL: Late subscribers get the last published value immediately.
                     Essential for parameters, maps, or any "current state" topic.
    VOLATILE: Late subscribers only get messages published after they connect.
              Fine for streaming data where old values are useless.
    """
    TRANSIENT_LOCAL = "transient_local"
    VOLATILE = "volatile"


@dataclass
class QoSProfile:
    """Quality of Service configuration for a publisher or subscriber.

    The key insight: publisher and subscriber QoS must be COMPATIBLE.
    A RELIABLE subscriber can't connect to a BEST_EFFORT publisher —
    the subscriber demands guaranteed delivery but the publisher won't provide it.
    A BEST_EFFORT subscriber CAN connect to a RELIABLE publisher — it just
    won't use the reliability guarantees.
    """
    reliability: Reliability = Reliability.RELIABLE
    durability: Durability = Durability.VOLATILE
    history_depth: int = 10  # How many messages to buffer

    @staticmethod
    def sensor_data() -> QoSProfile:
        """Profile for sensor streams: fast, lossy, no buffering."""
        return QoSProfile(
            reliability=Reliability.BEST_EFFORT,
            durability=Durability.VOLATILE,
            history_depth=1,
        )

    @staticmethod
    def reliable() -> QoSProfile:
        """Profile for commands/config: guaranteed delivery, some buffering."""
        return QoSProfile(
            reliability=Reliability.RELIABLE,
            durability=Durability.VOLATILE,
            history_depth=10,
        )

    @staticmethod
    def latched() -> QoSProfile:
        """Profile for state that late subscribers need: reliable + durable."""
        return QoSProfile(
            reliability=Reliability.RELIABLE,
            durability=Durability.TRANSIENT_LOCAL,
            history_depth=1,
        )


def qos_compatible(pub_qos: QoSProfile, sub_qos: QoSProfile) -> bool:
    """Check if publisher and subscriber QoS profiles are compatible.

    Compatibility rules (from the DDS spec):
    - Reliability: subscriber can't demand more than publisher offers
      RELIABLE pub + BEST_EFFORT sub = OK (sub just ignores guarantees)
      BEST_EFFORT pub + RELIABLE sub = INCOMPATIBLE
    - Durability: subscriber can't demand more than publisher offers
      TRANSIENT_LOCAL pub + VOLATILE sub = OK
      VOLATILE pub + TRANSIENT_LOCAL sub = INCOMPATIBLE
    """
    # Reliability check: RELIABLE > BEST_EFFORT in terms of guarantee level
    if (sub_qos.reliability == Reliability.RELIABLE and
            pub_qos.reliability == Reliability.BEST_EFFORT):
        return False

    # Durability check: TRANSIENT_LOCAL > VOLATILE in terms of guarantee level
    if (sub_qos.durability == Durability.TRANSIENT_LOCAL and
            pub_qos.durability == Durability.VOLATILE):
        return False

    return True


# =============================================================================
# Part 3: Topic Registry — The Communication Switchboard
# =============================================================================
# The registry is the middleware layer that decouples publishers from
# subscribers. In real ROS2, DDS handles this via multicast discovery.
# We simulate it with a global registry.

class TopicRegistry:
    """Central registry tracking all topics, publishers, and subscribers.

    This is our simplified version of the DDS discovery mechanism.
    In real ROS2, there's no central registry — nodes discover each other
    via multicast on the network. But the logical behavior is the same:
    when a publisher and subscriber both connect to the same topic name
    with compatible types and QoS, messages flow.
    """

    def __init__(self) -> None:
        # topic_name -> message type (for type checking)
        self._topic_types: Dict[str, Type[Message]] = {}
        # topic_name -> list of subscriber callbacks
        self._subscribers: Dict[str, List[Subscription]] = {}
        # topic_name -> last message (for TRANSIENT_LOCAL durability)
        self._latched: Dict[str, Message] = {}
        self._lock = threading.Lock()

    def register_topic(self, name: str, msg_type: Type[Message]) -> None:
        """Register a topic with its message type.

        If the topic already exists, verify the type matches — this catches
        the common bug where two nodes accidentally use the same topic name
        with different message types.
        """
        with self._lock:
            if name in self._topic_types:
                if self._topic_types[name] != msg_type:
                    raise TypeError(
                        f"Topic '{name}' already registered with type "
                        f"{self._topic_types[name].__name__}, "
                        f"cannot register with {msg_type.__name__}"
                    )
            else:
                self._topic_types[name] = msg_type
                self._subscribers.setdefault(name, [])

    def add_subscriber(self, topic: str, subscription: Subscription) -> None:
        """Add a subscriber to a topic."""
        with self._lock:
            self._subscribers.setdefault(topic, []).append(subscription)
            # If this topic has a latched message, deliver it immediately
            # by calling the callback directly (not queued — this happens
            # during connection setup, before the executor is involved).
            # This is how TRANSIENT_LOCAL works — late joiners get current state.
            if topic in self._latched:
                subscription.callback(self._latched[topic])

    def publish(self, topic: str, msg: Message, qos: QoSProfile) -> None:
        """Publish a message to all subscribers on a topic.

        This is where the actual message routing happens. Each subscriber
        gets a copy of the message delivered to its internal queue.
        """
        with self._lock:
            # Store for TRANSIENT_LOCAL durability
            if qos.durability == Durability.TRANSIENT_LOCAL:
                self._latched[topic] = msg

            for sub in self._subscribers.get(topic, []):
                # Check QoS compatibility before delivering
                if qos_compatible(qos, sub.qos):
                    sub.deliver(msg)

    def get_topic_names_and_types(self) -> Dict[str, str]:
        """List all registered topics and their types.

        Equivalent to `ros2 topic list -t` in a terminal.
        """
        with self._lock:
            return {
                name: typ.__name__
                for name, typ in self._topic_types.items()
            }


# Global registry — in real ROS2, this is the DDS participant
_registry = TopicRegistry()


def get_registry() -> TopicRegistry:
    """Get the global topic registry."""
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = TopicRegistry()


# =============================================================================
# Part 4: Publisher and Subscriber
# =============================================================================

class Publisher:
    """Publishes messages to a topic.

    A publisher is created by a node and bound to a specific topic.
    Each call to publish() sends the message to all compatible subscribers.

    Key design: the publisher doesn't know WHO is subscribing or HOW MANY.
    It just pushes to the topic. This is what makes the system modular —
    you can add/remove subscribers without touching publisher code.
    """

    def __init__(
        self,
        topic: str,
        msg_type: Type[Message],
        qos: QoSProfile,
        node_name: str,
    ) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.qos = qos
        self.node_name = node_name
        self._count = 0

        # Register the topic (validates type if already registered)
        registry = get_registry()
        registry.register_topic(topic, msg_type)

    def publish(self, msg: Message) -> None:
        """Publish a message to the topic.

        Type-checks the message at publish time — this catches bugs where
        a node accidentally publishes the wrong message type.
        """
        if not isinstance(msg, self.msg_type):
            raise TypeError(
                f"Publisher on '{self.topic}' expects {self.msg_type.__name__}, "
                f"got {type(msg).__name__}"
            )
        self._count += 1
        get_registry().publish(self.topic, msg, self.qos)

    @property
    def message_count(self) -> int:
        """How many messages this publisher has sent."""
        return self._count


class Subscription:
    """Subscribes to messages on a topic.

    When a message arrives, it's placed in an internal queue (not delivered
    immediately). The executor later drains the queue and calls the callback.
    This decouples message arrival from processing — critical because
    callbacks might take a while and we don't want to block the publisher.
    """

    def __init__(
        self,
        topic: str,
        msg_type: Type[Message],
        callback: Callable[[Message], None],
        qos: QoSProfile,
        node_name: str,
    ) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback
        self.qos = qos
        self.node_name = node_name
        # Message queue — bounded by QoS history depth to prevent memory blowup
        # In a real system, this would be a lock-free ring buffer
        self._queue: deque[Message] = deque(maxlen=qos.history_depth)
        self._lock = threading.Lock()

    def deliver(self, msg: Message) -> None:
        """Called by the registry to deliver a message to this subscriber.

        Thread-safe because publishers may be on different threads.
        If the queue is full, the oldest message is dropped (deque maxlen behavior).
        This is the right default — for sensor data, you always want the newest.
        """
        with self._lock:
            self._queue.append(msg)

    def take_messages(self) -> List[Message]:
        """Drain the queue and return all pending messages.

        Called by the executor during its spin cycle. Returns messages
        in FIFO order so callbacks process them chronologically.
        """
        with self._lock:
            msgs = list(self._queue)
            self._queue.clear()
            return msgs


# =============================================================================
# Part 5: Services — Request/Reply Pattern
# =============================================================================

@dataclass
class ServiceRequest(Message):
    """Base class for service requests."""
    pass


@dataclass
class ServiceResponse(Message):
    """Base class for service responses."""
    pass


@dataclass
class SetBoolRequest(ServiceRequest):
    """Request to set a boolean value (e.g., enable/disable a motor)."""
    value: bool = False


@dataclass
class SetBoolResponse(ServiceResponse):
    """Response confirming the boolean was set."""
    success: bool = False
    message: str = ""


@dataclass
class ComputePathRequest(ServiceRequest):
    """Request to compute a path between two points."""
    start_x: float = 0.0
    start_y: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0


@dataclass
class ComputePathResponse(ServiceResponse):
    """Response containing the computed path."""
    path: List[tuple] = field(default_factory=list)
    success: bool = False


class ServiceServer:
    """Handles incoming service requests.

    Unlike pub/sub (which is async and fire-and-forget), services are
    synchronous — the client blocks until the server responds.
    Use services for infrequent operations: parameter changes,
    mode switches, one-shot computations.
    """

    def __init__(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
        callback: Callable[[ServiceRequest], ServiceResponse],
    ) -> None:
        self.name = name
        self.request_type = request_type
        self.response_type = response_type
        self.callback = callback
        self._pending_requests: deque[tuple] = deque()
        self._lock = threading.Lock()

    def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a request and return the response.

        Type-checks both request and response to catch integration bugs.
        """
        if not isinstance(request, self.request_type):
            raise TypeError(
                f"Service '{self.name}' expects {self.request_type.__name__}, "
                f"got {type(request).__name__}"
            )
        response = self.callback(request)
        if not isinstance(response, self.response_type):
            raise TypeError(
                f"Service '{self.name}' callback must return "
                f"{self.response_type.__name__}, got {type(response).__name__}"
            )
        return response


class ServiceClient:
    """Sends requests to a service server.

    In real ROS2, the client would discover the server via DDS.
    Here we pass a direct reference for simplicity.
    """

    def __init__(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
    ) -> None:
        self.name = name
        self.request_type = request_type
        self.response_type = response_type
        self._server: Optional[ServiceServer] = None

    def connect(self, server: ServiceServer) -> None:
        """Connect to a service server."""
        if server.name != self.name:
            raise ValueError(
                f"Service name mismatch: client='{self.name}', server='{server.name}'"
            )
        self._server = server

    def call(self, request: ServiceRequest) -> ServiceResponse:
        """Send a request and block until the response arrives."""
        if self._server is None:
            raise RuntimeError(f"Service client '{self.name}' not connected to a server")
        return self._server.handle_request(request)


# =============================================================================
# Part 6: Timer — Periodic Callbacks
# =============================================================================

class Timer:
    """Triggers a callback at a fixed interval.

    Timers are how nodes do periodic work: publishing sensor data,
    running control loops, sending heartbeats. The executor checks
    each timer on every spin cycle and fires it if enough time has passed.

    Note: this is a "wall timer" (real time), not a "ROS timer" (sim time).
    In simulation, you'd use a clock topic to drive timers instead.
    """

    def __init__(self, period_seconds: float, callback: Callable[[], None]) -> None:
        self.period = period_seconds
        self.callback = callback
        self._last_fire = time.time()

    def ready(self) -> bool:
        """Check if enough time has passed since the last firing."""
        return (time.time() - self._last_fire) >= self.period

    def fire(self) -> None:
        """Execute the callback and reset the timer."""
        self._last_fire = time.time()
        self.callback()


# =============================================================================
# Part 7: Node — The Core Organizational Unit
# =============================================================================

class Node:
    """A ROS2 node — the fundamental unit of computation in a robot system.

    A node:
    - Has a unique name (used for debugging and introspection)
    - Creates publishers, subscribers, timers, and service servers/clients
    - Is "spun" by an executor that drives its callbacks

    Design principle: one node = one responsibility.
    A camera driver node publishes images. A detection node subscribes
    to images and publishes detections. Keep nodes focused and composable.
    """

    def __init__(self, name: str, namespace: str = "") -> None:
        self.name = name
        self.namespace = namespace
        self._publishers: List[Publisher] = []
        self._subscriptions: List[Subscription] = []
        self._timers: List[Timer] = []
        self._service_servers: List[ServiceServer] = []
        self._service_clients: List[ServiceClient] = []
        self._logger = NodeLogger(name)

    def _resolve_name(self, topic: str) -> str:
        """Resolve a topic name with the node's namespace.

        Namespacing prevents name collisions when running multiple instances
        of the same node. e.g., namespace='robot1' turns '/cmd_vel' into
        '/robot1/cmd_vel'.
        """
        if self.namespace and not topic.startswith("/"):
            return f"/{self.namespace}/{topic}"
        return topic

    def create_publisher(
        self,
        msg_type: Type[Message],
        topic: str,
        qos: Optional[QoSProfile] = None,
    ) -> Publisher:
        """Create a publisher on the given topic."""
        qos = qos or QoSProfile.reliable()
        resolved = self._resolve_name(topic)
        pub = Publisher(resolved, msg_type, qos, self.name)
        self._publishers.append(pub)
        self._logger.info(f"Created publisher on '{resolved}' [{msg_type.__name__}]")
        return pub

    def create_subscription(
        self,
        msg_type: Type[Message],
        topic: str,
        callback: Callable[[Message], None],
        qos: Optional[QoSProfile] = None,
    ) -> Subscription:
        """Create a subscription to the given topic."""
        qos = qos or QoSProfile.reliable()
        resolved = self._resolve_name(topic)
        # Register topic type for discovery
        get_registry().register_topic(resolved, msg_type)
        sub = Subscription(resolved, msg_type, callback, qos, self.name)
        get_registry().add_subscriber(resolved, sub)
        self._subscriptions.append(sub)
        self._logger.info(f"Created subscription on '{resolved}' [{msg_type.__name__}]")
        return sub

    def create_timer(self, period: float, callback: Callable[[], None]) -> Timer:
        """Create a periodic timer that fires every `period` seconds."""
        timer = Timer(period, callback)
        self._timers.append(timer)
        self._logger.info(f"Created timer with period {period:.3f}s")
        return timer

    def create_service(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
        callback: Callable[[ServiceRequest], ServiceResponse],
    ) -> ServiceServer:
        """Create a service server."""
        server = ServiceServer(name, request_type, response_type, callback)
        self._service_servers.append(server)
        self._logger.info(f"Created service server '{name}'")
        return server

    def create_client(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
    ) -> ServiceClient:
        """Create a service client."""
        client = ServiceClient(name, request_type, response_type)
        self._service_clients.append(client)
        self._logger.info(f"Created service client '{name}'")
        return client

    def get_logger(self) -> NodeLogger:
        """Get this node's logger."""
        return self._logger


class NodeLogger:
    """Simple logger that prefixes messages with the node name.

    In real ROS2, this integrates with the logging system and supports
    log levels, throttling, and output to both console and log files.
    """

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name

    def _log(self, level: str, msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] [{self.node_name}] {msg}")

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        self._log("WARN", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", msg)


# =============================================================================
# Part 8: Executor — The Event Loop
# =============================================================================

class SingleThreadedExecutor:
    """Drives node callbacks in a single-threaded event loop.

    The executor is the "heartbeat" of the system. Each spin cycle:
    1. Check all timers and fire any that are ready
    2. Drain all subscription queues and call callbacks

    Why single-threaded? Simplicity and safety. No mutex needed in callbacks,
    no race conditions, deterministic execution order. For most robots,
    a single thread running at 100+ Hz is more than enough.

    When you need more throughput, ROS2 offers MultiThreadedExecutor,
    but then every callback must be thread-safe.
    """

    def __init__(self) -> None:
        self._nodes: List[Node] = []

    def add_node(self, node: Node) -> None:
        """Add a node to be managed by this executor."""
        self._nodes.append(node)

    def spin_once(self) -> int:
        """Execute one cycle of the event loop.

        Returns the number of callbacks that were executed.
        This is useful for testing — you can step the system one tick at a time.
        """
        callback_count = 0

        for node in self._nodes:
            # Fire ready timers
            for timer in node._timers:
                if timer.ready():
                    timer.fire()
                    callback_count += 1

            # Process subscription messages
            for sub in node._subscriptions:
                for msg in sub.take_messages():
                    sub.callback(msg)
                    callback_count += 1

        return callback_count

    def spin(self, duration: float = 1.0, rate_hz: float = 100.0) -> None:
        """Spin the executor for a fixed duration.

        Args:
            duration: How long to spin (seconds)
            rate_hz: Target loop frequency (Hz)

        The loop sleeps to maintain the target rate. In a real system,
        you'd use a monotonic clock and handle overruns (when a spin
        cycle takes longer than the target period).
        """
        period = 1.0 / rate_hz
        start = time.time()

        while (time.time() - start) < duration:
            self.spin_once()
            time.sleep(period)


# =============================================================================
# Part 9: Demo — Multi-Node Robot System
# =============================================================================

def demo_basic_pubsub() -> None:
    """Demonstrate basic publish/subscribe communication."""
    print("=" * 70)
    print("DEMO 1: Basic Pub/Sub Communication")
    print("=" * 70)
    print()

    reset_registry()

    # Create a publisher node (simulates a sensor)
    sensor_node = Node("lidar_sensor")
    scan_pub = sensor_node.create_publisher(
        LaserScan, "/scan", QoSProfile.sensor_data()
    )
    print()

    # Create a subscriber node (simulates a controller)
    received_scans: List[LaserScan] = []

    def scan_callback(msg: LaserScan) -> None:
        min_range = min(msg.ranges) if msg.ranges else float('inf')
        sensor_node.get_logger().info(
            f"  -> Controller received scan: {len(msg.ranges)} rays, "
            f"min_range={min_range:.2f}m"
        )
        received_scans.append(msg)

    controller_node = Node("controller")
    controller_node.create_subscription(
        LaserScan, "/scan", scan_callback, QoSProfile.sensor_data()
    )
    print()

    # Publish some scans
    import math
    for i in range(3):
        # Simulate a scan with an obstacle at angle 0
        num_rays = 36  # 10-degree resolution
        ranges = []
        for j in range(num_rays):
            angle = -math.pi + j * (2 * math.pi / num_rays)
            # Obstacle at angle ~0 (straight ahead), distance 1.5m
            if abs(angle) < 0.2:
                ranges.append(1.5)
            else:
                ranges.append(10.0 + (j % 3) * 0.1)  # Far walls

        scan = LaserScan(ranges=ranges)
        scan_pub.publish(scan)
        sensor_node.get_logger().info(f"Published scan #{i + 1} with {num_rays} rays")

    # Process messages through executor
    executor = SingleThreadedExecutor()
    executor.add_node(sensor_node)
    executor.add_node(controller_node)
    executor.spin_once()

    print(f"\n  Publisher sent: {scan_pub.message_count} messages")
    print(f"  Subscriber received: {len(received_scans)} messages")
    print()


def demo_qos() -> None:
    """Demonstrate QoS compatibility and TRANSIENT_LOCAL durability."""
    print("=" * 70)
    print("DEMO 2: Quality of Service (QoS)")
    print("=" * 70)
    print()

    reset_registry()

    # --- QoS Compatibility ---
    print("  QoS Compatibility Matrix:")
    pub_profiles = [
        ("RELIABLE", QoSProfile(reliability=Reliability.RELIABLE)),
        ("BEST_EFFORT", QoSProfile(reliability=Reliability.BEST_EFFORT)),
    ]
    sub_profiles = [
        ("RELIABLE", QoSProfile(reliability=Reliability.RELIABLE)),
        ("BEST_EFFORT", QoSProfile(reliability=Reliability.BEST_EFFORT)),
    ]
    print(f"  {'':20s} | Sub RELIABLE | Sub BEST_EFFORT")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*15}")
    for pub_name, pub_qos in pub_profiles:
        row = f"  Pub {pub_name:14s} |"
        for _, sub_qos in sub_profiles:
            compat = qos_compatible(pub_qos, sub_qos)
            row += f" {'COMPAT':^12s} |" if compat else f" {'INCOMPAT':^12s} |"
        print(row)
    print()

    # --- TRANSIENT_LOCAL demonstration ---
    print("  TRANSIENT_LOCAL Durability Demo:")
    node_a = Node("map_server")
    map_pub = node_a.create_publisher(StringMsg, "/map", QoSProfile.latched())

    # Publish a map BEFORE any subscriber exists
    map_pub.publish(StringMsg(data="occupancy_grid_v1"))
    node_a.get_logger().info("Published map (before any subscriber)")

    # Now a late subscriber connects — it should get the map immediately
    received: List[StringMsg] = []

    def map_cb(msg: StringMsg) -> None:
        received.append(msg)

    node_b = Node("navigation")
    node_b.create_subscription(StringMsg, "/map", map_cb, QoSProfile.latched())
    node_b.get_logger().info(
        f"Late subscriber connected — received {len(received)} latched message(s)"
    )
    print(f"  -> Latched message data: '{received[0].data}'" if received else "")
    print()


def demo_services() -> None:
    """Demonstrate service request/reply pattern."""
    print("=" * 70)
    print("DEMO 3: Services (Request/Reply)")
    print("=" * 70)
    print()

    reset_registry()

    # Create a motor controller node with an enable/disable service
    motor_node = Node("motor_controller")
    motor_enabled = False

    def handle_set_motor(req: SetBoolRequest) -> SetBoolResponse:
        nonlocal motor_enabled
        motor_enabled = req.value
        motor_node.get_logger().info(
            f"Motor {'ENABLED' if req.value else 'DISABLED'}"
        )
        return SetBoolResponse(
            success=True,
            message=f"Motor set to {'enabled' if req.value else 'disabled'}",
        )

    motor_service = motor_node.create_service(
        "/set_motor", SetBoolRequest, SetBoolResponse, handle_set_motor
    )

    # Create a path planner node with a compute_path service
    planner_node = Node("path_planner")

    def handle_compute_path(req: ComputePathRequest) -> ComputePathResponse:
        # Simple straight-line path with 5 waypoints
        import math
        dx = req.goal_x - req.start_x
        dy = req.goal_y - req.start_y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        n_points = 5
        path = [
            (req.start_x + dx * i / (n_points - 1),
             req.start_y + dy * i / (n_points - 1))
            for i in range(n_points)
        ]
        planner_node.get_logger().info(
            f"Computed path: {n_points} waypoints, distance={dist:.2f}m"
        )
        return ComputePathResponse(path=path, success=True)

    path_service = planner_node.create_service(
        "/compute_path", ComputePathRequest, ComputePathResponse, handle_compute_path
    )

    # Client node calls both services
    client_node = Node("mission_controller")

    # Call motor service
    motor_client = client_node.create_client(
        "/set_motor", SetBoolRequest, SetBoolResponse
    )
    motor_client.connect(motor_service)
    print()

    response = motor_client.call(SetBoolRequest(value=True))
    client_node.get_logger().info(f"Motor service response: {response.message}")

    # Call path planning service
    path_client = client_node.create_client(
        "/compute_path", ComputePathRequest, ComputePathResponse
    )
    path_client.connect(path_service)

    response = path_client.call(ComputePathRequest(
        start_x=0.0, start_y=0.0, goal_x=5.0, goal_y=3.0
    ))
    client_node.get_logger().info(
        f"Path service response: success={response.success}, "
        f"waypoints={len(response.path)}"
    )
    for i, (x, y) in enumerate(response.path):
        print(f"    Waypoint {i}: ({x:.2f}, {y:.2f})")
    print()


def demo_full_system() -> None:
    """Demonstrate a complete multi-node robot system with timers.

    System architecture:
      [lidar_node] --/scan--> [controller_node] --/cmd_vel--> [motor_node]
                                                      |
                                              [logger_node] (listens to both)
    """
    print("=" * 70)
    print("DEMO 4: Full Robot System (Timers + Multi-Node)")
    print("=" * 70)
    print()

    reset_registry()
    import math

    # --- LIDAR Node ---
    # Publishes laser scans at regular intervals
    lidar_node = Node("lidar_driver")
    scan_pub = lidar_node.create_publisher(
        LaserScan, "/scan", QoSProfile.sensor_data()
    )
    scan_count = [0]

    def lidar_timer_cb() -> None:
        scan_count[0] += 1
        num_rays = 36
        ranges = []
        for j in range(num_rays):
            angle = -math.pi + j * (2 * math.pi / num_rays)
            # Simulate obstacle getting closer over time
            base_dist = max(0.5, 5.0 - scan_count[0] * 0.3)
            if abs(angle) < 0.3:
                ranges.append(base_dist)
            else:
                ranges.append(10.0)
        scan = LaserScan(ranges=ranges)
        scan_pub.publish(scan)

    lidar_node.create_timer(0.05, lidar_timer_cb)  # 20 Hz
    print()

    # --- Controller Node ---
    # Subscribes to /scan, publishes /cmd_vel
    controller_node = Node("reactive_controller")
    cmd_pub = controller_node.create_publisher(
        Twist, "/cmd_vel", QoSProfile.reliable()
    )

    def scan_callback(msg: LaserScan) -> None:
        if not msg.ranges:
            return
        min_range = min(msg.ranges)
        min_idx = msg.ranges.index(min_range)
        num_rays = len(msg.ranges)

        # Simple reactive controller:
        # - If obstacle is close, slow down and turn away
        # - Otherwise, go straight
        if min_range < 2.0:
            # Turn away from obstacle
            turn_direction = 1.0 if min_idx < num_rays // 2 else -1.0
            linear = max(0.0, (min_range - 0.5) * 0.5)  # Slow down near obstacles
            angular = turn_direction * (2.0 - min_range) * 0.5  # Turn harder when closer
            controller_node.get_logger().info(
                f"OBSTACLE at {min_range:.2f}m! -> vel=({linear:.2f}, {angular:.2f})"
            )
        else:
            linear = 1.0  # Full speed ahead
            angular = 0.0
            controller_node.get_logger().info(
                f"Clear path -> vel=({linear:.2f}, {angular:.2f})"
            )

        cmd_pub.publish(Twist(linear_x=linear, angular_z=angular))

    controller_node.create_subscription(
        LaserScan, "/scan", scan_callback, QoSProfile.sensor_data()
    )

    # --- Logger Node ---
    # Subscribes to everything for telemetry
    logger_node = Node("telemetry_logger")
    log_entries: List[str] = []

    def log_cmd(msg: Twist) -> None:
        entry = f"CMD: linear={msg.linear_x:.2f} angular={msg.angular_z:.2f}"
        log_entries.append(entry)

    logger_node.create_subscription(
        Twist, "/cmd_vel", log_cmd, QoSProfile.reliable()
    )
    print()

    # --- Run the system ---
    print("  Running system for 0.5 seconds...")
    print()
    executor = SingleThreadedExecutor()
    executor.add_node(lidar_node)
    executor.add_node(controller_node)
    executor.add_node(logger_node)
    executor.spin(duration=0.5, rate_hz=50.0)

    # --- Print system introspection ---
    print()
    print("  System Introspection:")
    topics = get_registry().get_topic_names_and_types()
    for topic_name, msg_type in topics.items():
        print(f"    Topic: {topic_name} [{msg_type}]")
    print(f"    LIDAR scans published: {scan_pub.message_count}")
    print(f"    Velocity commands logged: {len(log_entries)}")
    if log_entries:
        print(f"    Last command: {log_entries[-1]}")
    print()


if __name__ == "__main__":
    print()
    print("ROS2 Basic Node — Pure Python Implementation")
    print("=" * 70)
    print()
    print("This demo shows the core ROS2 communication primitives:")
    print("  1. Pub/Sub — asynchronous message passing between nodes")
    print("  2. QoS — reliability and durability guarantees")
    print("  3. Services — synchronous request/reply")
    print("  4. Full system — multi-node robot with timers and executor")
    print()

    demo_basic_pubsub()
    demo_qos()
    demo_services()
    demo_full_system()

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
  1. NODES are independent processes — each does one thing well
  2. TOPICS decouple publishers from subscribers — neither knows the other
  3. QoS lets you tune reliability vs latency per-topic
  4. SERVICES handle synchronous request/reply for one-shot operations
  5. The EXECUTOR drives everything — it's the event loop that fires callbacks
  6. This architecture scales from a toy robot to a self-driving car
    """)
