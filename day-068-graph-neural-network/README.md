# Day 068: Graph Neural Network (GNN) from Scratch

## What You're Building

A Graph Neural Network — the class of neural networks designed to operate on graph-structured data. While CNNs exploit spatial locality in grids and RNNs exploit sequential order, GNNs exploit the topology of arbitrary graphs: social networks, molecular structures, knowledge graphs, blockchain transaction networks, and robot scene graphs.

This matters because most real-world data isn't neatly arranged in grids or sequences. Molecules are graphs of atoms and bonds. Social networks are graphs of users and connections. Blockchain transactions form directed graphs. Scene understanding for robots involves spatial relationship graphs. GNNs let neural networks reason over these structures directly, rather than forcing graph data into a flat vector and losing structural information.

We'll implement a Graph Convolutional Network (GCN) following the Kipf & Welling (2017) formulation, then build a complete node classification system on a citation network.

## Core Concepts

### Why Graphs Need Special Neural Networks

Standard neural networks expect fixed-size inputs: a 224x224 image, a 512-dimensional vector. Graphs violate this assumption in every way:
- **Variable neighborhood size**: Node A might connect to 2 neighbors, node B to 200.
- **No canonical ordering**: Unlike pixels in an image, there's no "first" neighbor.
- **Permutation invariance**: Reordering the nodes shouldn't change the output.

The key insight of GNNs is **message passing**: each node updates its representation by aggregating information from its neighbors. After k rounds of message passing, each node's representation captures information from its k-hop neighborhood.

### The Graph Convolution Operation

In a standard convolution, we aggregate information from spatially adjacent pixels using learned weights. A graph convolution generalizes this: we aggregate information from topologically adjacent nodes.

Given:
- **Feature matrix** X ∈ ℝ^(N×F) — N nodes, each with F features
- **Adjacency matrix** A ∈ ℝ^(N×N) — A[i,j] = 1 if nodes i and j are connected
- **Weight matrix** W ∈ ℝ^(F×F') — learnable parameters mapping F input features to F' output features

The naive graph convolution is: H = σ(A · X · W)

But this has two critical problems:

**Problem 1: Missing self-loops.** When we multiply A·X, node i's new representation is the sum of its neighbors' features — but NOT its own features. We fix this by adding the identity matrix: Ã = A + I_N (adding self-connections).

**Problem 2: Scale explosion.** Nodes with many neighbors accumulate large values after aggregation, while nodes with few neighbors stay small. This creates numerical instability and makes features incomparable across nodes with different degrees.

The fix is **symmetric normalization**. Let D̃ be the degree matrix of Ã (diagonal matrix where D̃[i,i] = sum of row i of Ã). The normalized adjacency is:

**Â = D̃^(-1/2) · Ã · D̃^(-1/2)**

This means each neighbor's contribution is weighted by 1/√(deg(i) · deg(j)), which:
- Normalizes by the source node's degree (so high-degree nodes don't dominate)
- Normalizes by the target node's degree (so aggregated features stay bounded)
- Is symmetric, preserving the graph's undirected structure

The full GCN layer becomes: **H^(l+1) = σ(Â · H^(l) · W^(l))**

### Stacking Layers = Expanding Receptive Field

Each GCN layer lets each node see one hop further into the graph:
- **1 layer**: Each node knows about its direct neighbors
- **2 layers**: Each node knows about neighbors-of-neighbors  
- **k layers**: Each node has information from its k-hop neighborhood

But more layers isn't always better. GNNs suffer from **over-smoothing**: after too many layers, all node representations converge to the same value because every node has aggregated information from the entire graph. In practice, 2-3 layers works best for most tasks.

### Node Classification

The task: given a graph where some nodes have labels, predict labels for the unlabeled nodes. This is **semi-supervised learning** — we use both the graph structure AND a small set of labels.

For a citation network: nodes are papers, edges are citations, features are word vectors of the abstract, and labels are the paper's research area. A GCN can predict a paper's topic by looking at what it cites and what cites it.

## Step-by-Step Breakdown

### Step 1: Graph Representation
Store the graph as an adjacency matrix and feature matrix. Compute the normalized adjacency Â = D̃^(-1/2) · Ã · D̃^(-1/2) once upfront since the graph structure doesn't change during training.

### Step 2: GCN Layer (Forward Pass)
Each layer computes H' = σ(Â · H · W + b). The matrix multiply Â · H is the "message passing" — it replaces each node's features with a weighted average of its neighborhood. Then H · W is the learned linear transformation, and σ is a nonlinearity (ReLU for hidden layers, softmax for the output layer).

### Step 3: Multi-Layer GCN
Stack 2 GCN layers: the first maps input features to a hidden dimension with ReLU, the second maps hidden to output classes with softmax. Apply dropout between layers for regularization.

### Step 4: Training with Cross-Entropy Loss
Compute cross-entropy loss ONLY on labeled (training) nodes. Backpropagate through the entire graph — even though we only compute loss on some nodes, the gradients flow through the adjacency matrix multiplication, so unlabeled nodes' representations still get updated.

### Step 5: Backpropagation Through Graph Convolutions
The backward pass through Â · H · W requires careful chain rule application. The gradient with respect to W involves Â · H transposed. The gradient with respect to H (for propagating to earlier layers) involves Â transposed — which equals Â since our normalization is symmetric.

## Learning Objectives

- Understand message passing as the core GNN primitive
- Implement symmetric normalization of adjacency matrices and why it's necessary
- Build forward and backward passes for graph convolution layers
- Train a semi-supervised node classifier using only graph structure and a few labels
- Understand over-smoothing and the depth-accuracy tradeoff in GNNs

## Going Deeper

- **GraphSAGE**: Instead of using the full adjacency matrix (which requires all nodes in memory), sample a fixed number of neighbors per node. This enables mini-batch training on billion-node graphs.
- **Graph Attention Networks (GAT)**: Replace the fixed normalization weights (1/√(deg(i)·deg(j))) with learned attention weights, so the model can learn which neighbors are more important.
- **Spectral vs. Spatial**: GCN is derived from spectral graph theory (graph Fourier transforms), but the spatial interpretation (message passing) is more intuitive and generalizes better.
- **Applications**: Drug discovery (molecular property prediction), fraud detection (transaction graphs), recommendation systems (user-item graphs), traffic prediction (road networks), and combinatorial optimization (graph coloring, TSP).
- **Connection to Transformers**: Self-attention in transformers is equivalent to a GNN on a fully-connected graph. GNNs on sparse graphs are much more efficient.
