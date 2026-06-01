# Week 9 Recap

**May 27 - Jun 02, 2026** | 16 challenges completed | 15128 lines of code

---

## Challenges by Track

### AI

- **Day 046: Build a RAG pipeline** (Fri May 29)
  - 977 lines of code
- **Day 047: Embeddings and vector search** (Fri May 29)
  - 1157 lines of code
- **Day 048: Tool-using LLM agent** (Fri May 29)
  - 1227 lines of code
- **Day 049: Fine-tuning sentiment classifier** (Fri May 29)
  - 1267 lines of code
- **Day 050: Multi-agent conversation system** (Fri May 29)
  - 928 lines of code

### Crypto

- **Day 036: Hardhat testing setup** (Wed May 27)
  - 1675 lines of code
- **Day 037: DEX swap contract** (Wed May 27)
  - 932 lines of code
- **Day 038: Staking contract** (Wed May 27)
  - 720 lines of code
- **Day 039: Multisig wallet** (Wed May 27)
  - 961 lines of code
- **Day 040: Flash loan basics** (Wed May 27)
  - 781 lines of code
- **Day 051: Automated Market Maker (Constant Product)** (Sat May 30)
  - 790 lines of code

### Robotics

- **Day 041: Maze solver with BFS/DFS** (Thu May 28)
  - 572 lines of code
- **Day 042: Robot arm trajectory planning** (Thu May 28)
  - 768 lines of code
- **Day 043: Swarm behavior simulation** (Thu May 28)
  - 963 lines of code
- **Day 044: SLAM concept implementation** (Thu May 28)
  - 740 lines of code
- **Day 045: Kalman filter basics** (Thu May 28)
  - 670 lines of code

## Key Concepts Covered

### AI

- Understand why RAG exists and when to use it vs. fine-tuning vs. long-context
- Implement text chunking with overlap and understand the tradeoffs
- Generate and use text embeddings for semantic search
- Build a vector similarity search from scratch
- Construct effective RAG prompts with retrieved context
- Understand what embeddings represent geometrically and how they encode semantic meaning
- Implement cosine similarity, Euclidean distance, and dot product from scratch
- Build TF-IDF vectorization to understand sparse text representations
- Use sentence transformers to generate dense embeddings
- Implement LSH (Locality-Sensitive Hashing) for approximate nearest neighbor search
- Understand the ReAct agent loop and why it enables multi-step reasoning
- Build a tool registry with schema validation
- Parse structured actions from LLM output
- Implement context management for multi-turn agent conversations
- Design observable, debuggable agent systems
- Understand transfer learning and why pre-trained models are so effective
- Implement a complete fine-tuning pipeline: data prep → training → evaluation → inference
- Learn the critical hyperparameters and their effects (learning rate, epochs, batch size)
- Build intuition for overfitting detection in fine-tuning
- Understand tokenization and sequence handling for transformer models
- Design agent abstractions with role specialization
- Implement multiple conversation topologies (round-robin, coordinator)
- Build conversation state management with shared message history
- Handle termination conditions to prevent infinite loops
- Understand the tradeoffs between different multi-agent architectures

### Crypto

- Understand EVM state transitions and how testing frameworks simulate them
- Master ABI encoding/decoding at the byte level
- Learn test isolation patterns (snapshot/revert) specific to blockchain
- Build assertion utilities for events, reverts, and state changes
- Understand gas accounting and its role in testing
- Implement the constant product AMM formula and understand its mathematical properties
- Build liquidity provision/withdrawal with LP token accounting
- Calculate and visualize price impact, slippage, and impermanent loss
- Implement trading fees and understand how they accrue to LPs
- Add slippage protection (minimum output amounts)
- Implement the Synthetix reward accumulator pattern — the most widely-used staking math in DeFi
- Understand time-weighted reward distribution and why naive approaches don't scale
- Practice the checks-effects-interactions pattern for state safety
- Build intuition for reward rate calculations and period management
- Learn how staking contracts handle edge cases: zero total supply, mid-period reward additions, partial withdrawals
- Implement M-of-N threshold approval logic with proper access control
- Understand transaction lifecycle management (propose → confirm → execute)
- Apply the self-call pattern for governance-protected admin operations
- Handle edge cases: duplicate confirmations, executed transactions, threshold changes
- Build production-grade guard modifiers for contract security
- Understand atomic transaction guarantees and how DeFi exploits them
- Implement the callback pattern used by real flash loan protocols
- Build a fee calculation system with basis point precision
- Model arbitrage profit/loss including fees and slippage
- Understand the security implications of calling untrusted code
- Constant product invariant: Understand xy=k deeply — why it works as a pricing mechanism, its mathematical properties, and its limitations
- DeFi economics: See how fees, impermanent loss, and arbitrage interact to create a self-sustaining market
- Numerical precision: Handle the practical challenges of financial math — rounding, minimum amounts, and overflow prevention
- LP token mechanics: Understand how ownership shares are tracked and how deposits/withdrawals work
- Price impact: Quantify the cost of trading against a finite-liquidity pool

### Robotics

- Implement BFS and DFS from scratch with correct visited-set management
- Understand why BFS guarantees shortest paths in unweighted graphs (and when that guarantee breaks)
- Understand the memory/optimality tradeoff between BFS and DFS
- Build grid-based maze representations used in real robotics navigation
- Reconstruct paths from parent pointers — a pattern used across all graph search algorithms
- Understand the difference between path planning and trajectory generation
- Implement inverse kinematics for a 2-link planar arm
- Build trapezoidal and cubic polynomial trajectory generators
- Handle multi-segment trajectories with velocity continuity at via points
- Validate trajectories against joint and velocity constraints
- Understand emergent behavior from local interaction rules
- Implement the Boids flocking algorithm with separation, alignment, and cohesion
- Build a spatial hash for efficient O(n) neighbor queries
- Tune multi-objective force weights for desired swarm behavior
- Measure swarm quality metrics (cohesion, alignment, collision avoidance)
- Understand the SLAM problem and why it's fundamental to autonomous robots
- Implement the Extended Kalman Filter for joint robot-landmark estimation
- Work with Jacobian matrices for nonlinear state estimation
- See how cross-correlations enable map-wide corrections from single observations
- Appreciate the computational tradeoffs (EKF-SLAM is O(N²) per update)
- Understand state-space representation of dynamical systems
- Implement the predict-update cycle of a Kalman filter from scratch using NumPy
- Gain intuition for the Kalman gain and how it balances model vs. sensor trust
- Visualize how the filter reduces uncertainty over time
- Track both observable (position) and hidden (velocity) state variables

## Learning Objectives

**Day 036 — Hardhat testing setup:**
- Understand EVM state transitions and how testing frameworks simulate them
- Master ABI encoding/decoding at the byte level
- Learn test isolation patterns (snapshot/revert) specific to blockchain
- Build assertion utilities for events, reverts, and state changes
- Understand gas accounting and its role in testing
- Practice testing DeFi-style contracts for edge cases and security

**Day 037 — DEX swap contract:**
- Implement the constant product AMM formula and understand its mathematical properties
- Build liquidity provision/withdrawal with LP token accounting
- Calculate and visualize price impact, slippage, and impermanent loss
- Implement trading fees and understand how they accrue to LPs
- Add slippage protection (minimum output amounts)
- Understand the economic incentives that keep AMMs functioning

**Day 038 — Staking contract:**
- Implement the Synthetix reward accumulator pattern — the most widely-used staking math in DeFi
- Understand time-weighted reward distribution and why naive approaches don't scale
- Practice the checks-effects-interactions pattern for state safety
- Build intuition for reward rate calculations and period management
- Learn how staking contracts handle edge cases: zero total supply, mid-period reward additions, partial withdrawals

**Day 039 — Multisig wallet:**
- Implement M-of-N threshold approval logic with proper access control
- Understand transaction lifecycle management (propose → confirm → execute)
- Apply the self-call pattern for governance-protected admin operations
- Handle edge cases: duplicate confirmations, executed transactions, threshold changes
- Build production-grade guard modifiers for contract security

**Day 040 — Flash loan basics:**
- Understand atomic transaction guarantees and how DeFi exploits them
- Implement the callback pattern used by real flash loan protocols
- Build a fee calculation system with basis point precision
- Model arbitrage profit/loss including fees and slippage
- Understand the security implications of calling untrusted code
- See how flash loans connect to MEV, liquidations, and DeFi composability

**Day 041 — Maze solver with BFS/DFS:**
- Implement BFS and DFS from scratch with correct visited-set management
- Understand why BFS guarantees shortest paths in unweighted graphs (and when that guarantee breaks)
- Understand the memory/optimality tradeoff between BFS and DFS
- Build grid-based maze representations used in real robotics navigation
- Reconstruct paths from parent pointers — a pattern used across all graph search algorithms
- Develop intuition for when to use which algorithm through empirical comparison

**Day 042 — Robot arm trajectory planning:**
- Understand the difference between path planning and trajectory generation
- Implement inverse kinematics for a 2-link planar arm
- Build trapezoidal and cubic polynomial trajectory generators
- Handle multi-segment trajectories with velocity continuity at via points
- Validate trajectories against joint and velocity constraints

**Day 043 — Swarm behavior simulation:**
- Understand emergent behavior from local interaction rules
- Implement the Boids flocking algorithm with separation, alignment, and cohesion
- Build a spatial hash for efficient O(n) neighbor queries
- Tune multi-objective force weights for desired swarm behavior
- Measure swarm quality metrics (cohesion, alignment, collision avoidance)
- Connect to real applications: drone swarms, warehouse robots, search-and-rescue

**Day 044 — SLAM concept implementation:**
- Understand the SLAM problem and why it's fundamental to autonomous robots
- Implement the Extended Kalman Filter for joint robot-landmark estimation
- Work with Jacobian matrices for nonlinear state estimation
- See how cross-correlations enable map-wide corrections from single observations
- Appreciate the computational tradeoffs (EKF-SLAM is O(N²) per update)

**Day 045 — Kalman filter basics:**
- Understand state-space representation of dynamical systems
- Implement the predict-update cycle of a Kalman filter from scratch using NumPy
- Gain intuition for the Kalman gain and how it balances model vs. sensor trust
- Visualize how the filter reduces uncertainty over time
- Track both observable (position) and hidden (velocity) state variables

**Day 046 — Build a RAG pipeline:**
- Understand why RAG exists and when to use it vs. fine-tuning vs. long-context
- Implement text chunking with overlap and understand the tradeoffs
- Generate and use text embeddings for semantic search
- Build a vector similarity search from scratch
- Construct effective RAG prompts with retrieved context
- Add source attribution for answer provenance

**Day 047 — Embeddings and vector search:**
- Understand what embeddings represent geometrically and how they encode semantic meaning
- Implement cosine similarity, Euclidean distance, and dot product from scratch
- Build TF-IDF vectorization to understand sparse text representations
- Use sentence transformers to generate dense embeddings
- Implement LSH (Locality-Sensitive Hashing) for approximate nearest neighbor search
- Measure and compare search quality (recall@k) vs. speed tradeoffs
- Connect these foundations to production vector databases (FAISS, Pinecone, Weaviate)

**Day 048 — Tool-using LLM agent:**
- Understand the ReAct agent loop and why it enables multi-step reasoning
- Build a tool registry with schema validation
- Parse structured actions from LLM output
- Implement context management for multi-turn agent conversations
- Design observable, debuggable agent systems
- Connect this to production patterns (LangChain, Claude tool use, OpenAI function calling)

**Day 049 — Fine-tuning sentiment classifier:**
- Understand transfer learning and why pre-trained models are so effective
- Implement a complete fine-tuning pipeline: data prep → training → evaluation → inference
- Learn the critical hyperparameters and their effects (learning rate, epochs, batch size)
- Build intuition for overfitting detection in fine-tuning
- Understand tokenization and sequence handling for transformer models

**Day 050 — Multi-agent conversation system:**
- Design agent abstractions with role specialization
- Implement multiple conversation topologies (round-robin, coordinator)
- Build conversation state management with shared message history
- Handle termination conditions to prevent infinite loops
- Understand the tradeoffs between different multi-agent architectures
- See how multi-agent patterns connect to distributed systems concepts

**Day 051 — Automated Market Maker (Constant Product):**
- Constant product invariant: Understand xy=k deeply — why it works as a pricing mechanism, its mathematical properties, and its limitations
- DeFi economics: See how fees, impermanent loss, and arbitrage interact to create a self-sustaining market
- Numerical precision: Handle the practical challenges of financial math — rounding, minimum amounts, and overflow prevention
- LP token mechanics: Understand how ownership shares are tracked and how deposits/withdrawals work
- Price impact: Quantify the cost of trading against a finite-liquidity pool

## Cross-Track Connections

- Crypto and AI both explore concepts related to: sha
- Robotics and AI both explore concepts related to: hash

## Stats

| Metric | Value |
|--------|-------|
| Challenges completed | 16 |
| Total lines of code | 15128 |
| AI challenges | 5 |
| Crypto challenges | 6 |
| Robotics challenges | 5 |
