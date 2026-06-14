"""
Day 74: ROS2 Basic Node — Your Implementation

Build the core ROS2 communication primitives from scratch:
nodes, topics, publishers, subscribers, services, and an executor.

Hint: Think about WHY robotics uses pub/sub instead of direct function calls.
The key insight is DECOUPLING — publishers and subscribers don't know about each other.
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
# Hint: Messages are typed data containers. The base class carries a timestamp.
# Think about why timestamps matter for sensor fusion.

@dataclass
class Message:
    """Base class for all ROS2 messages."""
    timestamp: float = field(default_factory=time.time)


@dataclass
class StringMsg(Message):
    """Simple string message."""
    data: str = ""


@dataclass
class Twist(Message):
    """Velocity command — linear and angular velocities.

    For a differential-drive robot, you mainly use:
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
    """LIDAR scan data — ranges is a list of distances for each ray."""
    angle_min: float = -3.14159
    angle_max: float = 3.14159
    angle_increment: float = 0.01745
    range_min: float = 0.1
    range_max: float = 30.0
    ranges: List[float] = field(default_factory=list)


@dataclass
class Odometry(Message):
    """Robot position and velocity estimate."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


# =============================================================================
# Part 2: Quality of Service (QoS)
# =============================================================================
# Hint: QoS controls HOW messages are delivered.
# Key tradeoff: RELIABLE = guaranteed but slower, BEST_EFFORT = fast but lossy.

class Reliability(Enum):
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class Durability(Enum):
    TRANSIENT_LOCAL = "transient_local"
    VOLATILE = "volatile"


@dataclass
class QoSProfile:
    """Quality of Service configuration.

    Hint: Think about compatibility rules.
    A RELIABLE subscriber can't work with a BEST_EFFORT publisher
    (it demands guarantees the publisher won't provide).
    """
    reliability: Reliability = Reliability.RELIABLE
    durability: Durability = Durability.VOLATILE
    history_depth: int = 10

    @staticmethod
    def sensor_data() -> QoSProfile:
        """Profile for sensor streams."""
        return QoSProfile(
            reliability=Reliability.BEST_EFFORT,
            durability=Durability.VOLATILE,
            history_depth=1,
        )

    @staticmethod
    def reliable() -> QoSProfile:
        """Profile for commands."""
        return QoSProfile(
            reliability=Reliability.RELIABLE,
            durability=Durability.VOLATILE,
            history_depth=10,
        )

    @staticmethod
    def latched() -> QoSProfile:
        """Profile for state that late subscribers need."""
        return QoSProfile(
            reliability=Reliability.RELIABLE,
            durability=Durability.TRANSIENT_LOCAL,
            history_depth=1,
        )


def qos_compatible(pub_qos: QoSProfile, sub_qos: QoSProfile) -> bool:
    """Check if publisher and subscriber QoS profiles are compatible.

    Hint: The subscriber can't demand MORE than the publisher offers.
    RELIABLE > BEST_EFFORT, TRANSIENT_LOCAL > VOLATILE.
    """
    raise NotImplementedError("TODO: implement QoS compatibility check")


# =============================================================================
# Part 3: Topic Registry
# =============================================================================
# Hint: The registry is the "switchboard" that routes messages from
# publishers to subscribers. Think about type safety — what happens if
# two nodes use the same topic name with different message types?

class TopicRegistry:
    """Central registry tracking all topics, publishers, and subscribers."""

    def __init__(self) -> None:
        raise NotImplementedError("TODO: initialize the registry data structures")

    def register_topic(self, name: str, msg_type: Type[Message]) -> None:
        """Register a topic with its message type.

        Hint: If the topic already exists, verify the type matches.
        Raise TypeError on mismatch.
        """
        raise NotImplementedError("TODO: implement topic registration")

    def add_subscriber(self, topic: str, subscription: Subscription) -> None:
        """Add a subscriber to a topic.

        Hint: Check for latched messages (TRANSIENT_LOCAL) and call
        the subscriber's callback directly (not queued) for immediate delivery.
        """
        raise NotImplementedError("TODO: implement subscriber registration")

    def publish(self, topic: str, msg: Message, qos: QoSProfile) -> None:
        """Publish a message to all subscribers on a topic.

        Hint: Check QoS compatibility before delivering.
        Store message for TRANSIENT_LOCAL durability.
        """
        raise NotImplementedError("TODO: implement message publishing")

    def get_topic_names_and_types(self) -> Dict[str, str]:
        """List all registered topics and their types."""
        raise NotImplementedError("TODO: implement topic listing")


# Global registry
_registry = TopicRegistry.__new__(TopicRegistry)


def get_registry() -> TopicRegistry:
    """Get the global topic registry."""
    return _registry


def reset_registry() -> None:
    """Reset the global registry."""
    global _registry
    _registry = TopicRegistry()


# =============================================================================
# Part 4: Publisher and Subscriber
# =============================================================================

class Publisher:
    """Publishes messages to a topic.

    Hint: The publisher doesn't know who subscribes. It just pushes
    messages to the registry. Type-check messages at publish time.
    """

    def __init__(
        self,
        topic: str,
        msg_type: Type[Message],
        qos: QoSProfile,
        node_name: str,
    ) -> None:
        raise NotImplementedError("TODO: implement publisher initialization")

    def publish(self, msg: Message) -> None:
        """Publish a message to the topic.

        Hint: Type-check the message against self.msg_type.
        """
        raise NotImplementedError("TODO: implement message publishing")

    @property
    def message_count(self) -> int:
        """How many messages this publisher has sent."""
        raise NotImplementedError("TODO: implement message count")


class Subscription:
    """Subscribes to messages on a topic.

    Hint: Messages go into a bounded queue (deque with maxlen).
    The executor later drains the queue and calls the callback.
    """

    def __init__(
        self,
        topic: str,
        msg_type: Type[Message],
        callback: Callable[[Message], None],
        qos: QoSProfile,
        node_name: str,
    ) -> None:
        raise NotImplementedError("TODO: implement subscription initialization")

    def deliver(self, msg: Message) -> None:
        """Called by the registry to deliver a message.

        Hint: Thread-safe! Use a lock. Add to the bounded deque.
        """
        raise NotImplementedError("TODO: implement message delivery")

    def take_messages(self) -> List[Message]:
        """Drain the queue and return all pending messages.

        Hint: Return in FIFO order. Clear the queue after draining.
        """
        raise NotImplementedError("TODO: implement message draining")


# =============================================================================
# Part 5: Services
# =============================================================================
# Hint: Services are synchronous request/reply — unlike pub/sub which is async.
# Use services for infrequent operations (config changes, one-shot computations).

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
    value: bool = False


@dataclass
class SetBoolResponse(ServiceResponse):
    success: bool = False
    message: str = ""


@dataclass
class ComputePathRequest(ServiceRequest):
    start_x: float = 0.0
    start_y: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0


@dataclass
class ComputePathResponse(ServiceResponse):
    path: List[tuple] = field(default_factory=list)
    success: bool = False


class ServiceServer:
    """Handles incoming service requests."""

    def __init__(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
        callback: Callable[[ServiceRequest], ServiceResponse],
    ) -> None:
        raise NotImplementedError("TODO: implement service server initialization")

    def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a request and return the response.

        Hint: Type-check both request and response.
        """
        raise NotImplementedError("TODO: implement request handling")


class ServiceClient:
    """Sends requests to a service server."""

    def __init__(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
    ) -> None:
        raise NotImplementedError("TODO: implement service client initialization")

    def connect(self, server: ServiceServer) -> None:
        """Connect to a service server.

        Hint: Verify the service names match.
        """
        raise NotImplementedError("TODO: implement server connection")

    def call(self, request: ServiceRequest) -> ServiceResponse:
        """Send a request and block until the response arrives."""
        raise NotImplementedError("TODO: implement service call")


# =============================================================================
# Part 6: Timer
# =============================================================================

class Timer:
    """Triggers a callback at a fixed interval.

    Hint: Track the last fire time. ready() checks if enough
    time has passed since the last firing.
    """

    def __init__(self, period_seconds: float, callback: Callable[[], None]) -> None:
        raise NotImplementedError("TODO: implement timer initialization")

    def ready(self) -> bool:
        """Check if enough time has passed since the last firing."""
        raise NotImplementedError("TODO: implement readiness check")

    def fire(self) -> None:
        """Execute the callback and reset the timer."""
        raise NotImplementedError("TODO: implement timer firing")


# =============================================================================
# Part 7: Node
# =============================================================================

class Node:
    """A ROS2 node — the fundamental unit of computation.

    Hint: A node creates publishers, subscribers, timers, and services.
    It has a name and namespace. The namespace prefixes topic names
    to prevent collisions.
    """

    def __init__(self, name: str, namespace: str = "") -> None:
        raise NotImplementedError("TODO: implement node initialization")

    def _resolve_name(self, topic: str) -> str:
        """Resolve a topic name with the node's namespace.

        Hint: If namespace is 'robot1' and topic is 'cmd_vel',
        return '/robot1/cmd_vel'. If topic starts with '/', don't add namespace.
        """
        raise NotImplementedError("TODO: implement name resolution")

    def create_publisher(
        self,
        msg_type: Type[Message],
        topic: str,
        qos: Optional[QoSProfile] = None,
    ) -> Publisher:
        """Create a publisher on the given topic."""
        raise NotImplementedError("TODO: implement publisher creation")

    def create_subscription(
        self,
        msg_type: Type[Message],
        topic: str,
        callback: Callable[[Message], None],
        qos: Optional[QoSProfile] = None,
    ) -> Subscription:
        """Create a subscription to the given topic."""
        raise NotImplementedError("TODO: implement subscription creation")

    def create_timer(self, period: float, callback: Callable[[], None]) -> Timer:
        """Create a periodic timer."""
        raise NotImplementedError("TODO: implement timer creation")

    def create_service(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
        callback: Callable[[ServiceRequest], ServiceResponse],
    ) -> ServiceServer:
        """Create a service server."""
        raise NotImplementedError("TODO: implement service server creation")

    def create_client(
        self,
        name: str,
        request_type: Type[ServiceRequest],
        response_type: Type[ServiceResponse],
    ) -> ServiceClient:
        """Create a service client."""
        raise NotImplementedError("TODO: implement service client creation")

    def get_logger(self) -> NodeLogger:
        """Get this node's logger."""
        raise NotImplementedError("TODO: implement logger access")


class NodeLogger:
    """Simple logger prefixed with node name."""

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
# Part 8: Executor
# =============================================================================

class SingleThreadedExecutor:
    """Drives node callbacks in a single-threaded event loop.

    Hint: Each spin cycle:
    1. Check all timers and fire ready ones
    2. Drain all subscription queues and call callbacks
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: implement executor initialization")

    def add_node(self, node: Node) -> None:
        """Add a node to be managed by this executor."""
        raise NotImplementedError("TODO: implement node addition")

    def spin_once(self) -> int:
        """Execute one cycle of the event loop.

        Returns the number of callbacks executed.
        """
        raise NotImplementedError("TODO: implement single spin cycle")

    def spin(self, duration: float = 1.0, rate_hz: float = 100.0) -> None:
        """Spin the executor for a fixed duration at the given rate."""
        raise NotImplementedError("TODO: implement timed spin loop")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    import math

    print("Testing ROS2 Basic Node Implementation")
    print("=" * 50)

    # Test 1: Basic pub/sub
    print("\nTest 1: Basic Pub/Sub")
    reset_registry()
    node_a = Node("publisher_node")
    node_b = Node("subscriber_node")

    pub = node_a.create_publisher(StringMsg, "/chatter")
    received = []
    node_b.create_subscription(StringMsg, "/chatter", lambda m: received.append(m))

    pub.publish(StringMsg(data="Hello ROS2!"))

    executor = SingleThreadedExecutor()
    executor.add_node(node_a)
    executor.add_node(node_b)
    executor.spin_once()

    assert len(received) == 1, f"Expected 1 message, got {len(received)}"
    assert received[0].data == "Hello ROS2!"
    print("  PASSED!")

    # Test 2: QoS compatibility
    print("\nTest 2: QoS Compatibility")
    assert qos_compatible(QoSProfile.reliable(), QoSProfile.sensor_data()) == True
    assert qos_compatible(QoSProfile.sensor_data(), QoSProfile.reliable()) == False
    print("  PASSED!")

    # Test 3: Services
    print("\nTest 3: Services")
    reset_registry()
    server_node = Node("server")
    srv = server_node.create_service(
        "/test_srv", SetBoolRequest, SetBoolResponse,
        lambda req: SetBoolResponse(success=True, message=f"Set to {req.value}")
    )
    client_node = Node("client")
    cli = client_node.create_client("/test_srv", SetBoolRequest, SetBoolResponse)
    cli.connect(srv)
    resp = cli.call(SetBoolRequest(value=True))
    assert resp.success == True
    print("  PASSED!")

    # Test 4: Timer
    print("\nTest 4: Timer")
    count = [0]
    timer = Timer(0.01, lambda: count.__setitem__(0, count[0] + 1))
    time.sleep(0.02)
    assert timer.ready()
    timer.fire()
    assert count[0] == 1
    print("  PASSED!")

    print("\nAll tests passed!")
