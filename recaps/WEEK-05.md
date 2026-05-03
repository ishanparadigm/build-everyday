# Week 5 Recap

**Apr 29 - May 05, 2026** | 5 challenges completed | 4435 lines of code

---

## Challenges by Track

### Crypto

- **Day 029: Hello World Solidity contract** (Sat May 02)
  - 1085 lines of code
- **Day 030: ERC-20 token implementation** (Sat May 02)
  - 668 lines of code
- **Day 031: ERC-721 NFT contract** (Sat May 02)
  - 806 lines of code

### Robotics

- **Day 028: A* pathfinding** (Thu Apr 30)
  - 655 lines of code

### Integration

- **Day 032: Smart contract that uses oracle ML predictions** (Sun May 03)
  - 1221 lines of code

## Key Concepts Covered

### Crypto

- Understand how smart contracts execute at the VM level, not just the syntax level
- Implement function dispatch using selector hashing (the mechanism behind Solidity's ABI)
- Build a gas metering system that enforces computational limits and reverts on exhaustion
- Implement the storage model (slot-based key-value) that underlies all Solidity state variables
- Understand events as cheap, write-only, off-chain-indexed data structures
- Understand the ERC-20 interface and why each function exists
- Implement the approve/transferFrom delegation pattern
- Handle fixed-point token arithmetic with decimals
- Recognize the approval race condition vulnerability
- Build minting/burning with supply tracking
- Understand the ERC-721 standard and how non-fungible ownership works on-chain
- Implement dual-layer approval mechanics (per-token and operator)
- Build safe transfer patterns with receiver callbacks
- Handle minting, burning, and metadata management
- Learn event-driven architecture for off-chain indexing

### Robotics

- Implement A search with correct open/closed set management
- Understand how heuristic choice affects search behavior and optimality
- Compare A exploration patterns against BFS and Dijkstra
- Handle 4-directional and 8-directional movement with appropriate heuristics
- Reconstruct optimal paths using parent pointers

### Integration

- Understand the oracle problem and why blockchains need external data bridges
- Implement commit-reveal schemes to prevent front-running
- Integrate off-chain ML predictions with on-chain smart contract logic
- Handle fixed-point arithmetic (basis points) for on-chain financial math
- Design staleness checks and liveness guarantees for oracle data

## Learning Objectives

**Day 028 — A* pathfinding:**
- Implement A search with correct open/closed set management
- Understand how heuristic choice affects search behavior and optimality
- Compare A exploration patterns against BFS and Dijkstra
- Handle 4-directional and 8-directional movement with appropriate heuristics
- Reconstruct optimal paths using parent pointers
- Analyze time and space complexity of informed search

**Day 029 — Hello World Solidity contract:**
- Understand how smart contracts execute at the VM level, not just the syntax level
- Implement function dispatch using selector hashing (the mechanism behind Solidity's ABI)
- Build a gas metering system that enforces computational limits and reverts on exhaustion
- Implement the storage model (slot-based key-value) that underlies all Solidity state variables
- Understand events as cheap, write-only, off-chain-indexed data structures
- Implement owner-based access control — the pattern behind OpenZeppelin's Ownable
- See how deployment differs from execution and why constructors only run once

**Day 030 — ERC-20 token implementation:**
- Understand the ERC-20 interface and why each function exists
- Implement the approve/transferFrom delegation pattern
- Handle fixed-point token arithmetic with decimals
- Recognize the approval race condition vulnerability
- Build minting/burning with supply tracking
- Design a clean state machine for financial token logic

**Day 031 — ERC-721 NFT contract:**
- Understand the ERC-721 standard and how non-fungible ownership works on-chain
- Implement dual-layer approval mechanics (per-token and operator)
- Build safe transfer patterns with receiver callbacks
- Handle minting, burning, and metadata management
- Learn event-driven architecture for off-chain indexing
- Practice defensive programming with ownership and authorization checks

**Day 032 — Smart contract that uses oracle ML predictions:**
- Understand the oracle problem and why blockchains need external data bridges
- Implement commit-reveal schemes to prevent front-running
- Integrate off-chain ML predictions with on-chain smart contract logic
- Handle fixed-point arithmetic (basis points) for on-chain financial math
- Design staleness checks and liveness guarantees for oracle data
- Build a multi-oracle aggregation system with median consensus
- Connect ML model outputs to real contract decisions (lending, interest rates)

## Stats

| Metric | Value |
|--------|-------|
| Challenges completed | 5 |
| Total lines of code | 4435 |
| Crypto challenges | 3 |
| Robotics challenges | 1 |
| Integration challenges | 1 |
