# Day 011: Robot That Follows ML-Detected Objects

## Integration Challenge: AI + Robotics

### Overview

Build a simulated robot that uses a machine learning object detector to identify and follow a target in a 2D environment. This challenge integrates two tracks you've been building skills in: **AI** (classification, distance metrics) and **Robotics** (PID control, kinematics, state machines).

In the real world, this is the foundation of autonomous systems everywhere: warehouse robots following inventory carts, delivery drones tracking landing pads, agricultural robots following crop rows, and self-driving cars tracking lane markers. The core loop is always the same: **perceive** (detect the object), **decide** (compute where to go), and **act** (send motor commands).

### Core Concepts

#### 1. The Perception-Action Loop

Every autonomous robot runs a continuous cycle:

```
Sense -> Perceive -> Plan -> Act -> Sense -> ...
```

- **Sense**: Raw sensor data (camera pixels, lidar points, etc.)
- **Perceive**: Extract meaning from raw data (object detection, classification)
- **Plan**: Decide what to do (compute heading, speed)
- **Act**: Send commands to actuators (motors, servos)

The key challenge is that each stage introduces **latency** and **noise**. Your detector might misclassify objects. Your position estimate drifts. Your motors don't respond instantly. Robust systems must handle all of this gracefully.

#### 2. Simplified Object Detection

In production, you'd use a CNN like YOLO or SSD. Here we simulate detection with a model that:

1. Takes a "scene" (list of objects with positions, types, and features)
2. Classifies each object using a simple ML classifier (KNN from Day 009)
3. Returns bounding boxes and class labels with simulated noise

The key insight: **detection is classification applied spatially**. Each region of an image gets classified, and regions with high confidence become detections. We simulate this by adding position noise and occasional false positives/negatives.

#### 3. Target Tracking with Filtering

Raw detections are noisy. Frame-to-frame, the detected position jumps around even if the target moves smoothly. We use an **Exponential Moving Average (EMA)** filter to smooth detections:

```
smoothed_position = alpha * new_detection + (1 - alpha) * previous_smoothed
```

Where `alpha` controls responsiveness vs. smoothness:
- `alpha` close to 1.0: very responsive but noisy
- `alpha` close to 0.0: very smooth but laggy

This is a simplified version of Kalman filtering. The tradeoff between responsiveness and stability is fundamental to all real-time control systems.

#### 4. PID-Based Following (from Day 006)

Once we know where the target is, we need to drive toward it. We use two PID controllers:

- **Heading PID**: Controls the robot's angular velocity to point at the target
- **Distance PID**: Controls the robot's linear velocity to maintain a follow distance

The heading controller computes:
```
angle_error = atan2(target_y - robot_y, target_x - robot_x) - robot_heading
angular_velocity = Kp * error + Ki * integral(error) + Kd * d(error)/dt
```

The distance controller computes:
```
distance_error = distance_to_target - desired_follow_distance
linear_velocity = Kp * error + Ki * integral(error) + Kd * d(error)/dt
```

#### 5. State Machine for Behavior (from Day 007)

The robot doesn't just blindly chase. It has behavioral states:

- **SEARCHING**: No target detected. Robot rotates in place scanning.
- **ACQUIRING**: Target detected but not yet confirmed. Wait for N consecutive detections to filter out false positives.
- **FOLLOWING**: Confirmed target. PID controllers active, robot follows.
- **LOST**: Target was being followed but detection lost. Robot slows down and enters a grace period before going back to SEARCHING.

This state machine prevents jittery behavior from detection noise and gives the system robustness.

### Step-by-Step Breakdown

1. **Build the simulated environment**: A 2D world with a robot and multiple objects. Objects have types (target, obstacle, distractor) and move along predefined paths. The robot has position, heading, and velocity.

2. **Implement the ML detector**: Use KNN to classify objects based on their features (color, size, shape encoded as numbers). Add realistic noise: position jitter, missed detections (false negatives), and phantom detections (false positives).

3. **Build the tracking filter**: EMA filter that smooths noisy detections. Maintains a confidence score that decays when detections are missed and grows when detections are consistent.

4. **Implement PID following**: Two PID controllers (heading and distance) that generate robot velocity commands. Include velocity clamping and angle wrapping for correctness.

5. **Wire up the state machine**: States (SEARCHING, ACQUIRING, FOLLOWING, LOST) with transitions driven by detection confidence and tracking status.

6. **Run the simulation loop**: Discrete time steps. Each step: move objects, run detector, update tracker, run state machine, apply PID, move robot. Log everything for analysis.

7. **Analyze performance**: Compute tracking metrics — average follow distance error, time in each state, number of target losses, and reaction time to target movements.

### Learning Objectives

- Integrate ML classification into a real-time control loop
- Understand the perception-action cycle in autonomous systems
- Apply PID control for target following (heading + distance)
- Use state machines for robust behavior management
- Handle noisy, imperfect sensor data with filtering
- Analyze end-to-end system performance with meaningful metrics

### Going Deeper

- **Kalman Filter**: Replace EMA with a proper Kalman filter that models the target's velocity for predictive tracking (see Day 008's forward kinematics for related math)
- **Multi-Object Tracking**: Track multiple targets simultaneously using Hungarian algorithm for detection-to-track assignment
- **Occlusion Handling**: What happens when the target goes behind an obstacle? Predict where it will reappear using motion models
- **Deep Learning Detection**: Replace KNN with a CNN — the architecture changes but the perception-action loop stays identical
- **Hardware Deployment**: This exact architecture runs on ROS2 with camera nodes publishing detections and motor nodes subscribing to velocity commands
- **Adversarial Robustness**: What if objects try to fool the classifier? This connects to adversarial ML research
