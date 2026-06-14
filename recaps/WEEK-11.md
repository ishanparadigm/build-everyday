# Week 11 Recap

**Jun 10 - Jun 16, 2026** | 5 challenges completed | 6234 lines of code

---

## Challenges by Track

### Crypto

- **Day 072: On-chain data analysis with Python** (Fri Jun 12)
  - 1284 lines of code
- **Day 073: MEV detection script** (Sat Jun 13)
  - 1246 lines of code

### Robotics

- **Day 070: Autonomous drone path planning** (Wed Jun 10)
  - 1182 lines of code
- **Day 071: Reinforcement learning for robot control** (Thu Jun 11)
  - 1004 lines of code
- **Day 074: ROS2 basic node** (Sun Jun 14)
  - 1518 lines of code

## Key Concepts Covered

### Crypto

- Parse and index blockchain transaction data structures efficiently
- Profile addresses by behavioral patterns using statistical features
- Build and analyze directed weighted graphs from token flow data
- Apply time-series anomaly detection to on-chain metrics
- Compute distribution inequality metrics (Gini coefficient)
- Understand MEV taxonomy: sandwich attacks, arbitrage, liquidations, and JIT liquidity
- Learn transaction ordering economics and why mempool visibility creates extraction opportunities
- Implement pattern-matching algorithms for detecting structured transaction sequences
- Analyze gas price dynamics as MEV signals
- Understand the game theory between searchers, builders, and validators

### Robotics

- Implement sampling-based path planning in 3D (extending Day 34's RRT to RRT)
- Design physics-informed cost functions beyond simple distance
- Handle complex 3D collision detection with multiple obstacle types
- Apply path smoothing for real-world flyable trajectories
- Understand the tradeoffs between planning time, path quality, and computational cost
- Connect RL theory to robotics practice: See how MDPs, value functions, and policy learning apply to physical control
- Understand reward engineering: Experience the tradeoffs in reward shaping for continuous control
- Implement DQN from scratch: Build all components — replay buffer, target network, ε-schedule — understanding why each exists
- Appreciate sim-to-real challenges: Understand why simulation fidelity and domain randomization matter
- Understand publish/subscribe as a communication pattern and why robotics chose it over RPC
- Implement typed message passing with QoS guarantees
- Build a working executor/event loop that drives asynchronous node communication
- Learn the node-topic-message architecture that underpins all ROS2 systems
- Understand services as the complement to pub/sub for synchronous operations

## Learning Objectives

**Day 070 — Autonomous drone path planning:**
- Implement sampling-based path planning in 3D (extending Day 34's RRT to RRT)
- Design physics-informed cost functions beyond simple distance
- Handle complex 3D collision detection with multiple obstacle types
- Apply path smoothing for real-world flyable trajectories
- Understand the tradeoffs between planning time, path quality, and computational cost

**Day 071 — Reinforcement learning for robot control:**
- Connect RL theory to robotics practice: See how MDPs, value functions, and policy learning apply to physical control
- Understand reward engineering: Experience the tradeoffs in reward shaping for continuous control
- Implement DQN from scratch: Build all components — replay buffer, target network, ε-schedule — understanding why each exists
- Appreciate sim-to-real challenges: Understand why simulation fidelity and domain randomization matter

**Day 072 — On-chain data analysis with Python:**
- Parse and index blockchain transaction data structures efficiently
- Profile addresses by behavioral patterns using statistical features
- Build and analyze directed weighted graphs from token flow data
- Apply time-series anomaly detection to on-chain metrics
- Compute distribution inequality metrics (Gini coefficient)
- Understand how real-world on-chain analytics platforms work under the hood

**Day 073 — MEV detection script:**
- Understand MEV taxonomy: sandwich attacks, arbitrage, liquidations, and JIT liquidity
- Learn transaction ordering economics and why mempool visibility creates extraction opportunities
- Implement pattern-matching algorithms for detecting structured transaction sequences
- Analyze gas price dynamics as MEV signals
- Understand the game theory between searchers, builders, and validators

**Day 074 — ROS2 basic node:**
- Understand publish/subscribe as a communication pattern and why robotics chose it over RPC
- Implement typed message passing with QoS guarantees
- Build a working executor/event loop that drives asynchronous node communication
- Learn the node-topic-message architecture that underpins all ROS2 systems
- Understand services as the complement to pub/sub for synchronous operations
- Reason about QoS tradeoffs between reliability and latency

## Stats

| Metric | Value |
|--------|-------|
| Challenges completed | 5 |
| Total lines of code | 6234 |
| Crypto challenges | 2 |
| Robotics challenges | 3 |
