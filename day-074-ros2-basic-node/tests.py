"""
Day 74: ROS2 Basic Node — Test Suite

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import unittest
import time
import math
from my_solution import (
    Message, StringMsg, Twist, LaserScan, Odometry,
    Reliability, Durability, QoSProfile, qos_compatible,
    TopicRegistry, Publisher, Subscription,
    ServiceServer, ServiceClient,
    ServiceRequest, ServiceResponse,
    SetBoolRequest, SetBoolResponse,
    ComputePathRequest, ComputePathResponse,
    Timer, Node, SingleThreadedExecutor,
    get_registry, reset_registry,
)


class TestMessages(unittest.TestCase):
    """Test message creation and typing."""

    def test_string_msg(self):
        msg = StringMsg(data="hello")
        self.assertEqual(msg.data, "hello")
        self.assertIsInstance(msg, Message)
        self.assertIsInstance(msg.timestamp, float)

    def test_twist_defaults(self):
        twist = Twist()
        self.assertEqual(twist.linear_x, 0.0)
        self.assertEqual(twist.angular_z, 0.0)

    def test_laser_scan(self):
        scan = LaserScan(ranges=[1.0, 2.0, 3.0])
        self.assertEqual(len(scan.ranges), 3)
        self.assertAlmostEqual(scan.angle_min, -math.pi, places=3)


class TestQoS(unittest.TestCase):
    """Test QoS profiles and compatibility."""

    def test_reliable_pub_reliable_sub(self):
        """RELIABLE pub + RELIABLE sub = compatible."""
        self.assertTrue(qos_compatible(QoSProfile.reliable(), QoSProfile.reliable()))

    def test_reliable_pub_best_effort_sub(self):
        """RELIABLE pub + BEST_EFFORT sub = compatible (sub ignores guarantees)."""
        self.assertTrue(qos_compatible(
            QoSProfile.reliable(), QoSProfile.sensor_data()
        ))

    def test_best_effort_pub_reliable_sub(self):
        """BEST_EFFORT pub + RELIABLE sub = INCOMPATIBLE."""
        self.assertFalse(qos_compatible(
            QoSProfile.sensor_data(), QoSProfile.reliable()
        ))

    def test_volatile_pub_transient_local_sub(self):
        """VOLATILE pub + TRANSIENT_LOCAL sub = INCOMPATIBLE."""
        pub_qos = QoSProfile(durability=Durability.VOLATILE)
        sub_qos = QoSProfile(durability=Durability.TRANSIENT_LOCAL)
        self.assertFalse(qos_compatible(pub_qos, sub_qos))

    def test_transient_local_pub_volatile_sub(self):
        """TRANSIENT_LOCAL pub + VOLATILE sub = compatible."""
        pub_qos = QoSProfile(durability=Durability.TRANSIENT_LOCAL)
        sub_qos = QoSProfile(durability=Durability.VOLATILE)
        self.assertTrue(qos_compatible(pub_qos, sub_qos))


class TestPubSub(unittest.TestCase):
    """Test publisher/subscriber communication."""

    def setUp(self):
        reset_registry()

    def test_basic_pubsub(self):
        """Publisher sends, subscriber receives via executor."""
        node_a = Node("pub_node")
        node_b = Node("sub_node")

        pub = node_a.create_publisher(StringMsg, "/test_topic")
        received = []
        node_b.create_subscription(
            StringMsg, "/test_topic", lambda m: received.append(m)
        )

        pub.publish(StringMsg(data="test"))

        executor = SingleThreadedExecutor()
        executor.add_node(node_a)
        executor.add_node(node_b)
        executor.spin_once()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data, "test")

    def test_multiple_subscribers(self):
        """Multiple subscribers all receive the same message."""
        node = Node("pub")
        pub = node.create_publisher(StringMsg, "/multi")

        received_a, received_b = [], []
        sub_a = Node("sub_a")
        sub_b = Node("sub_b")
        sub_a.create_subscription(StringMsg, "/multi", lambda m: received_a.append(m))
        sub_b.create_subscription(StringMsg, "/multi", lambda m: received_b.append(m))

        pub.publish(StringMsg(data="broadcast"))

        executor = SingleThreadedExecutor()
        executor.add_node(sub_a)
        executor.add_node(sub_b)
        executor.spin_once()

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    def test_type_mismatch_publish(self):
        """Publishing wrong type should raise TypeError."""
        node = Node("node")
        pub = node.create_publisher(StringMsg, "/typed")
        with self.assertRaises(TypeError):
            pub.publish(Twist())

    def test_type_mismatch_topic(self):
        """Registering same topic with different type should raise TypeError."""
        node_a = Node("a")
        node_a.create_publisher(StringMsg, "/conflict")
        node_b = Node("b")
        with self.assertRaises(TypeError):
            node_b.create_publisher(Twist, "/conflict")

    def test_message_count(self):
        """Publisher tracks message count."""
        node = Node("counter")
        pub = node.create_publisher(StringMsg, "/count")
        self.assertEqual(pub.message_count, 0)
        pub.publish(StringMsg(data="a"))
        pub.publish(StringMsg(data="b"))
        self.assertEqual(pub.message_count, 2)


class TestTransientLocal(unittest.TestCase):
    """Test TRANSIENT_LOCAL durability (latched messages)."""

    def setUp(self):
        reset_registry()

    def test_late_subscriber_gets_latched(self):
        """A subscriber connecting after publish gets the latched message."""
        node_a = Node("pub")
        pub = node_a.create_publisher(StringMsg, "/latched", QoSProfile.latched())

        # Publish BEFORE subscriber exists
        pub.publish(StringMsg(data="initial_state"))

        # Now subscribe
        received = []
        node_b = Node("sub")
        node_b.create_subscription(
            StringMsg, "/latched", lambda m: received.append(m),
            QoSProfile.latched()
        )

        # The latched message should be delivered at subscription time
        # (before any spin)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data, "initial_state")


class TestServices(unittest.TestCase):
    """Test service request/reply pattern."""

    def setUp(self):
        reset_registry()

    def test_basic_service(self):
        """Service server processes request and returns response."""
        node = Node("srv_node")
        srv = node.create_service(
            "/enable", SetBoolRequest, SetBoolResponse,
            lambda req: SetBoolResponse(success=True, message="done")
        )

        client_node = Node("client")
        cli = client_node.create_client("/enable", SetBoolRequest, SetBoolResponse)
        cli.connect(srv)

        resp = cli.call(SetBoolRequest(value=True))
        self.assertTrue(resp.success)
        self.assertEqual(resp.message, "done")

    def test_service_name_mismatch(self):
        """Connecting to wrong service name should raise ValueError."""
        node = Node("srv")
        srv = node.create_service(
            "/svc_a", SetBoolRequest, SetBoolResponse,
            lambda r: SetBoolResponse()
        )
        cli_node = Node("cli")
        cli = cli_node.create_client("/svc_b", SetBoolRequest, SetBoolResponse)
        with self.assertRaises(ValueError):
            cli.connect(srv)

    def test_unconnected_client(self):
        """Calling without connecting should raise RuntimeError."""
        node = Node("cli")
        cli = node.create_client("/missing", SetBoolRequest, SetBoolResponse)
        with self.assertRaises(RuntimeError):
            cli.call(SetBoolRequest())


class TestTimerAndExecutor(unittest.TestCase):
    """Test timer and executor functionality."""

    def test_timer_fires(self):
        """Timer fires when enough time has passed."""
        count = [0]
        timer = Timer(0.01, lambda: count.__setitem__(0, count[0] + 1))
        time.sleep(0.02)
        self.assertTrue(timer.ready())
        timer.fire()
        self.assertEqual(count[0], 1)
        # Should not be ready immediately after firing
        self.assertFalse(timer.ready())

    def test_executor_spin_once(self):
        """Executor processes pending messages in one spin."""
        reset_registry()
        node = Node("test")
        pub = node.create_publisher(StringMsg, "/exec_test")
        received = []
        node.create_subscription(
            StringMsg, "/exec_test", lambda m: received.append(m)
        )
        pub.publish(StringMsg(data="spin"))

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        callbacks = executor.spin_once()

        self.assertEqual(len(received), 1)
        self.assertGreater(callbacks, 0)


class TestNamespace(unittest.TestCase):
    """Test node namespace resolution."""

    def setUp(self):
        reset_registry()

    def test_namespace_prefix(self):
        """Node namespace prefixes topic names."""
        node = Node("sensor", namespace="robot1")
        pub = node.create_publisher(StringMsg, "data")
        # The resolved topic should be /robot1/data
        self.assertEqual(pub.topic, "/robot1/data")

    def test_absolute_topic_ignores_namespace(self):
        """Topics starting with / ignore namespace."""
        node = Node("sensor", namespace="robot1")
        pub = node.create_publisher(StringMsg, "/global_topic")
        self.assertEqual(pub.topic, "/global_topic")


if __name__ == "__main__":
    unittest.main()
