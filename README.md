# build-everyday

<!-- BADGES:START -->
![Days](https://img.shields.io/badge/days-33-blue) ![Streak](https://img.shields.io/badge/streak-1_days-orange) ![Longest](https://img.shields.io/badge/longest-5_days-green) ![Updated](https://img.shields.io/badge/last_updated-2026--05--06-lightgrey)
<!-- BADGES:END -->

Daily coding across AI, crypto, and robotics. One commit a day, every day.

## Structure

```
day-XXX-topic/
  README.md    -- challenge description
  solution.*   -- implementation
```

## Tracks

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|-----|-----|-----|-----|-----|-----|-----|-----|
| Track | AI | Crypto | Robotics | AI | Crypto | Robotics | Integration |

## Progression

**Weeks 1-4** — Fundamentals
- AI: ML from scratch (regression, trees, neural nets)
- Crypto: Primitives (hashing, signing, Merkle trees)
- Robotics: Control systems (PID, state machines)

**Weeks 5-8** — Building
- AI: LLM applications (prompt chains, RAG, agents)
- Crypto: Smart contracts (Solidity, testing, deployment)
- Robotics: Path planning & simulation (A*, RRT, PyBullet)

**Weeks 9-12** — Advanced
- AI: Computer vision, reinforcement learning
- Crypto: DeFi protocols, on-chain analytics
- Robotics: Sensor fusion, SLAM

**Week 13+** — Integration projects combining multiple tracks

<!-- HEATMAP:START -->
![Contribution Heatmap](docs/heatmap.svg)
<!-- HEATMAP:END -->

<!-- STREAK:START -->
### 🔥 Current Streak: 1 days | Longest: 5 days
<!-- STREAK:END -->

<!-- PROGRESS_BARS:START -->
**AI         ** `███████░░░░░░░░░░░░░ 8/24`  
**Crypto     ** `██████░░░░░░░░░░░░░░ 7/24`  
**Robotics   ** `████████░░░░░░░░░░░░ 10/24`  
**Integration** `████████░░░░░░░░░░░░ 5/12`  

**Total LOC**: 18,817 lines across 33 solutions
<!-- PROGRESS_BARS:END -->



## Dashboard

| Week | Mon (AI) | Tue (Crypto) | Wed (Robotics) | Thu (AI) | Fri (Crypto) | Sat (Robotics) | Sun (Integration) |
|------|----------|--------------|-----------------|----------|--------------|-----------------|-------------------|
| 1 | — | [001](day-001-linear-regression/) | [002](day-002-sha256-hash/) | [003](day-003-logistic-regression/) | — | [004](day-004-decision-tree/) | [005](day-005-kmeans-clustering/) |
| 2 | [006](day-006-pid-controller/) [007](day-007-state-machine/) | — | [008](day-008-kinematics/) | [009](day-009-knn-classifier/) | — | — | [010](day-010-crypto-price-predictor/) |
| 3 | [012](day-012-naive-bayes/) | [013](day-013-merkle-tree/) [020](day-020-digital-signatures/) | [014](day-014-motor-control/) [021](day-021-line-following-robot/) | [015](day-015-neural-network-forward/) | — | [016](day-016-sensor-simulator/) [017](day-017-obstacle-avoidance/) | [011](day-011-ml-object-follower/) [018](day-018-onchain-ml-registry/) [019](day-019-trading-bot-skeleton/) |
| 4 | — | [027](day-027-proof-of-work/) | [022](day-022-servo-control/) [028](day-028-astar-pathfinding/) | — | [029](day-029-hello-solidity/) [030](day-030-erc20-token/) [031](day-031-erc721-nft/) | — | [024](day-024-blockchain-sensor-integrity/) [025](day-025-anomaly-blockchain-audit/) [026](day-026-signed-robot-commands/) |
| 5 | [033](day-033-prompt-chaining/) | — | [034](day-034-rrt-path-planning/) | — | — | — | [032](day-032-oracle-ml-contract/) |

## Progress

- Day 001: [Linear Regression from Scratch](day-001-linear-regression/) - Normal equation and gradient descent implementations with feature scaling analysis
- Day 002: [SHA-256 Hash Implementation](day-002-sha256-hash/) - Complete SHA-256 from bitwise operations with avalanche effect analysis
- Day 003: [Logistic Regression from Scratch](day-003-logistic-regression/) - Binary classification with sigmoid, cross-entropy loss, and decision boundary analysis
- Day 004: [Decision Tree Classifier](day-004-decision-tree/) - CART algorithm with entropy/Gini splitting, depth-based regularization, and feature importance
- Day 005: [K-Means Clustering](day-005-kmeans-clustering/) - Unsupervised clustering with K-Means++ initialization, elbow analysis, and convergence visualization
- Day 006: [PID Controller Simulation](day-006-pid-controller/) - Proportional-integral-derivative control with thermal plant simulation, step response analysis, and tuning exploration
- Day 007: [State Machine for Robot Behavior](day-007-state-machine/) - Generic FSM engine with guard conditions, entry/exit actions, and patrol robot behavior simulation
- Day 008: [Forward and Inverse Kinematics](day-008-kinematics/) - 2D robotic arm kinematics with analytical IK, Jacobian pseudo-inverse numerical IK, and workspace analysis
- Day 009: [KNN from Scratch](day-009-knn-classifier/) - K-Nearest Neighbors with multiple distance metrics, weighted voting, cross-validation for k selection, and curse of dimensionality analysis
- Day 010: [AI-Powered Crypto Price Predictor](day-010-crypto-price-predictor/) - Integration of ML regression with crypto market data for price prediction
- Day 011: [Robot That Follows ML-Detected Objects](day-011-ml-object-follower/) - Integration of KNN detection, PID control, and state machines for autonomous target following
- Day 012: [Naive Bayes Classifier](day-012-naive-bayes/) - Multinomial Naive Bayes spam detector with Bayesian reasoning, log-space arithmetic, and Laplace smoothing
- Day 013: [Merkle Tree from Scratch](day-013-merkle-tree/) - Binary hash tree with O(log n) inclusion proofs, tamper detection, and domain-separated hashing
- Day 014: [Motor Control Simulation](day-014-motor-control/) - DC motor dynamics with RK4 integration, PWM speed control, and cascaded PID position/velocity controllers
- Day 015: [Simple Neural Network Forward Pass](day-015-neural-network-forward/) - Feedforward network with He initialization, ReLU/sigmoid/softmax activations, cross-entropy loss, and mini-batch processing
- Day 016: [Sensor Reading Simulator](day-016-sensor-simulator/) - Robotics sensor framework with LIDAR raycasting, IMU bias drift, odometry dead reckoning, and Gaussian sensor fusion
- Day 017: [Obstacle Avoidance Algorithm](day-017-obstacle-avoidance/) - Vector Field Histogram (VFH) with ray-cast sensors, polar histogram construction, valley detection, and cost-based steering
- Day 018: [On-Chain ML Model Registry](day-018-onchain-ml-registry/) - Blockchain-backed model registry with content-addressed identity, HMAC signatures, Merkle inclusion proofs, and tamper detection
- Day 019: [Autonomous Trading Bot Skeleton](day-019-trading-bot-skeleton/) - Event-driven trading bot with MA crossover + RSI signals, control-system risk management, and simulated execution with spread and slippage
- Day 020: [Digital Signatures (ECDSA Basics)](day-020-digital-signatures/) - ECDSA from scratch on secp256k1 with point arithmetic, key generation, sign/verify, and nonce-reuse attack demonstration
- Day 021: [Line-Following Robot Logic](day-021-line-following-robot/) - Differential-drive robot with reflectance sensor array, bang-bang/P/PID control comparison, and quantitative performance analysis
- Day 022: [Servo Control Patterns](day-022-servo-control/) - PWM signal generation, trapezoidal/easing motion profiles, multi-servo synchronization, and keyframe-based pick-and-place sequences
- Day 024: [Blockchain-Verified Sensor Data Pipeline](day-024-blockchain-sensor-integrity/) - Integration of robot sensor simulation, hash-chained integrity ledger, rolling z-score anomaly detection, and tamper-evidence verification
- Day 025: [AI Anomaly Detection with Blockchain Audit Trail](day-025-anomaly-blockchain-audit/) - Isolation Forest from scratch for unsupervised anomaly detection with SHA-256 hash-chained audit ledger and tamper-evidence verification
- Day 026: [Cryptographically Signed Robot Command Protocol](day-026-signed-robot-commands/) - ECDSA-signed commands with Merkle batch verification, state machine execution flow, PID-controlled movement, and attack resistance (replay, tampering, unauthorized access)
- Day 027: [Proof of Work Simulation](day-027-proof-of-work/) - SHA-256 mining loop with dynamic difficulty adjustment, chain validation, 51% attack Monte Carlo simulation, and exponential scaling analysis
- Day 028: [A* Pathfinding](day-028-astar-pathfinding/) - A* search on 2D grids with Manhattan/Euclidean/Octile/Chebyshev heuristics, 4-dir and 8-dir movement, corner-cutting prevention, and exploration efficiency comparison
- Day 029: [Hello World Solidity Contract](day-029-hello-solidity/) - Mini smart contract VM with slot-based storage, function selector dispatch, gas metering with cold/warm access costs, event emission, owner access control, and revert/rollback semantics
- Day 030: [ERC-20 Token Implementation](day-030-erc20-token/) - Full ERC-20 standard with approve/transferFrom delegation, mint/burn supply management, safe allowance helpers, event logging, and approval race condition analysis
- Day 031: [ERC-721 NFT Contract](day-031-erc721-nft/) - Full ERC-721 standard with per-token and operator approvals, safe transfer receiver callbacks, mint/burn, metadata URIs, and enumerable extension with O(1) swap-and-pop removal
- Day 032: [Smart Contract with Oracle ML Predictions](day-032-oracle-ml-contract/) - ML credit risk model feeding oracle contract with commit-reveal, median aggregation, staleness checks, and lending contract with piecewise interest rate curves and front-running prevention
- Day 033: [Prompt Chaining with Claude API](day-033-prompt-chaining/) - Multi-step LLM pipeline with sequential/parallel execution, JSON validation between steps, retry logic with error feedback, and per-step observability metrics
- Day 034: [RRT Path Planning](day-034-rrt-path-planning/) - Rapidly-exploring Random Trees with goal biasing, edge collision detection, path smoothing via shortcutting, and RRT* asymptotically optimal rewiring
