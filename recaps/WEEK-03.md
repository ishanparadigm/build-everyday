# Week 3 Recap

**Apr 15 - Apr 21, 2026** | 5 challenges completed | 4230 lines of code

---

## Challenges by Track

### AI

- **Day 015: Simple neural network (forward pass)** (Thu Apr 16)
  - 695 lines of code

### Robotics

- **Day 014: Motor control simulation** (Wed Apr 15)
  - 1017 lines of code
- **Day 016: Sensor reading simulator** (Sat Apr 18)
  - 976 lines of code
- **Day 017: Obstacle avoidance algorithm** (Sat Apr 18)
  - 771 lines of code
- **Day 017: Obstacle avoidance algorithm** (Sat Apr 18)
  - 771 lines of code

## Key Concepts Covered

### AI

- Implement a feedforward neural network using only NumPy
- Understand the forward pass as a sequence of linear transformations + nonlinearities
- Implement ReLU, sigmoid, and softmax activation functions with numerical stability
- Apply proper weight initialization (He, Xavier)
- Compute cross-entropy loss for multi-class classification

### Robotics

- Model electromechanical systems from first principles (coupled ODEs)
- Implement RK4 numerical integration for accurate simulation
- Understand PWM and its role in power electronics
- Build and tune cascaded PID controllers for velocity and position
- Analyze transient response: rise time, overshoot, settling time, steady-state error
- Understand Gaussian noise models and how sensor parameters affect measurement quality
- Implement realistic sensor simulations for LIDAR, IMU, and wheel encoders
- See how bias drift causes unbounded error growth in dead reckoning
- Apply basic sensor fusion to combine redundant measurements
- Build the foundation for Day 6's PID controller and future SLAM/Kalman filter work
- Understand reactive vs. deliberative navigation and when each is appropriate
- Implement the Vector Field Histogram algorithm from first principles
- Work with polar coordinates, angular arithmetic (wrapping!), and sensor models
- Handle noisy sensor data through aggregation and thresholding
- Design cost functions that balance competing objectives
- Understand reactive vs. deliberative navigation and when each is appropriate
- Implement the Vector Field Histogram algorithm from first principles
- Work with polar coordinates, angular arithmetic (wrapping!), and sensor models
- Handle noisy sensor data through aggregation and thresholding
- Design cost functions that balance competing objectives

## Learning Objectives

**Day 014 — Motor control simulation:**
- Model electromechanical systems from first principles (coupled ODEs)
- Implement RK4 numerical integration for accurate simulation
- Understand PWM and its role in power electronics
- Build and tune cascaded PID controllers for velocity and position
- Analyze transient response: rise time, overshoot, settling time, steady-state error

**Day 015 — Simple neural network (forward pass):**
- Implement a feedforward neural network using only NumPy
- Understand the forward pass as a sequence of linear transformations + nonlinearities
- Implement ReLU, sigmoid, and softmax activation functions with numerical stability
- Apply proper weight initialization (He, Xavier)
- Compute cross-entropy loss for multi-class classification
- Process data in mini-batches using matrix operations
- Build intuition for how network depth and width affect representational capacity

**Day 016 — Sensor reading simulator:**
- Understand Gaussian noise models and how sensor parameters affect measurement quality
- Implement realistic sensor simulations for LIDAR, IMU, and wheel encoders
- See how bias drift causes unbounded error growth in dead reckoning
- Apply basic sensor fusion to combine redundant measurements
- Build the foundation for Day 6's PID controller and future SLAM/Kalman filter work

**Day 017 — Obstacle avoidance algorithm:**
- Understand reactive vs. deliberative navigation and when each is appropriate
- Implement the Vector Field Histogram algorithm from first principles
- Work with polar coordinates, angular arithmetic (wrapping!), and sensor models
- Handle noisy sensor data through aggregation and thresholding
- Design cost functions that balance competing objectives
- Build a simulation loop that demonstrates emergent intelligent behavior from simple rules

**Day 017 — Obstacle avoidance algorithm:**
- Understand reactive vs. deliberative navigation and when each is appropriate
- Implement the Vector Field Histogram algorithm from first principles
- Work with polar coordinates, angular arithmetic (wrapping!), and sensor models
- Handle noisy sensor data through aggregation and thresholding
- Design cost functions that balance competing objectives
- Build a simulation loop that demonstrates emergent intelligent behavior from simple rules

## Stats

| Metric | Value |
|--------|-------|
| Challenges completed | 5 |
| Total lines of code | 4230 |
| AI challenges | 1 |
| Robotics challenges | 4 |
