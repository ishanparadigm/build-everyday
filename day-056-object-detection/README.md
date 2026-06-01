# Day 056: Object Detection Basics — Building a Single-Shot Detector from Scratch

## Overview

Object detection is the task of not just classifying *what* is in an image, but *where* each object is located. While image classification (Day 055) outputs a single label, object detection outputs a list of bounding boxes, each with a class label and confidence score. This is the backbone of self-driving cars, warehouse robotics, medical imaging, and surveillance systems.

Today we build a simplified single-shot detector (SSD) from scratch using only NumPy. We implement the core pipeline: anchor box generation, Intersection over Union (IoU), non-maximum suppression (NMS), and a basic detection head that maps feature maps to bounding box predictions.

## Core Concepts

### Bounding Boxes and Coordinate Representations

A bounding box is a rectangle that tightly encloses an object. Two common formats:

- **Corner format**: `(x_min, y_min, x_max, y_max)` — the top-left and bottom-right corners
- **Center format**: `(cx, cy, w, h)` — center coordinates plus width and height

Converting between them:
```
cx = (x_min + x_max) / 2,  cy = (y_min + y_max) / 2
w  = x_max - x_min,         h  = y_max - y_min
```

We'll work in corner format for IoU calculations and center format for anchor generation.

### Intersection over Union (IoU)

IoU measures how much two boxes overlap. It's the single most important metric in object detection:

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)
          = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

Computing the intersection:
```
x_min_inter = max(A.x_min, B.x_min)
y_min_inter = max(A.y_min, B.y_min)
x_max_inter = min(A.x_max, B.x_max)
y_max_inter = min(A.y_max, B.y_max)

If x_max_inter <= x_min_inter or y_max_inter <= y_min_inter:
    intersection = 0  (boxes don't overlap)
else:
    intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
```

**Why IoU and not just overlap area?** Raw overlap area is scale-dependent — a 100px overlap is great for two small boxes but terrible for two large ones. IoU normalizes to [0, 1], making it scale-invariant.

**IoU thresholds in practice:**
- IoU > 0.5: standard detection match (PASCAL VOC)
- IoU > 0.75: strict match (COCO "AP75")
- IoU > 0.5 used during NMS to merge redundant detections

### Anchor Boxes (Prior Boxes)

The key insight of modern detectors: don't predict bounding boxes from nothing — predict *adjustments* to a set of pre-defined reference boxes called **anchors**.

Why? The space of all possible (x, y, w, h) is huge. Anchors constrain the problem: instead of predicting absolute coordinates, the network predicts small deltas (offsets) from anchors. This makes learning much easier and faster to converge.

**Generating anchors:**
1. Tile a grid over the image (e.g., 8x8 cells for a 256x256 image)
2. At each grid cell, place anchors of multiple aspect ratios (e.g., 1:1, 1:2, 2:1) and scales
3. Each anchor has a center at the grid cell center and a specific width/height

For an 8x8 grid with 3 aspect ratios, you get 8 * 8 * 3 = 192 anchors. Real detectors use multiple feature map scales (SSD uses 6 scales), yielding thousands of anchors.

**Encoding offsets (box regression targets):**
```
tx = (gt_cx - anchor_cx) / anchor_w
ty = (gt_cy - anchor_cy) / anchor_h
tw = log(gt_w / anchor_w)
th = log(gt_h / anchor_h)
```

The log transform for width/height ensures the network predicts multiplicative scaling factors, which are more stable than additive offsets (a box can't have negative width).

### Non-Maximum Suppression (NMS)

Multiple anchors will fire for the same object. NMS removes redundant detections:

1. Sort all detections by confidence score (descending)
2. Take the highest-scoring detection, add it to the final list
3. Remove all remaining detections that have IoU > threshold with the selected one
4. Repeat until no detections remain

**Why greedy NMS works:** If two boxes have high IoU, they're likely detecting the same object. We keep the more confident one. The threshold (typically 0.5) controls the tradeoff between removing duplicates (too low = miss nearby objects) and keeping duplicates (too high = multiple boxes per object).

### The Detection Pipeline

Putting it all together:
1. **Generate anchors** across the image at multiple scales/ratios
2. **For each anchor**, predict: (a) class probabilities, (b) bounding box offsets
3. **Decode predictions**: apply predicted offsets to anchors to get actual box coordinates
4. **Filter**: remove low-confidence predictions (score < threshold)
5. **NMS**: remove duplicate detections per class

## Step-by-Step Breakdown

1. **Bounding box utilities**: Implement conversion functions and IoU computation. These are the atomic operations everything else depends on.

2. **Anchor generation**: Create a grid of anchor boxes with configurable scales and aspect ratios. This defines the "search space" of possible detections.

3. **Anchor matching**: Given ground-truth boxes, match each anchor to a ground-truth box using IoU. Anchors with IoU > 0.5 are "positive" (responsible for detecting that object); IoU < 0.3 are "negative" (background); in between are ignored during training.

4. **Offset encoding/decoding**: Compute the regression targets (offsets from anchor to ground-truth) for training, and the inverse (offsets back to absolute coordinates) for inference.

5. **NMS implementation**: The post-processing step that turns raw per-anchor predictions into clean, non-overlapping detections.

6. **Simulated detection pipeline**: Generate synthetic "ground truth" scenes, create fake model outputs with noise, and run the full decode + NMS pipeline to produce final detections.

## Learning Objectives

- Implement IoU computation and understand its role as the universal matching metric
- Build anchor box generation with multi-scale, multi-ratio configurations
- Implement non-maximum suppression from scratch
- Understand the encode/decode cycle for bounding box regression
- See how all components connect in an end-to-end detection pipeline

## Going Deeper

- **Multi-scale detection**: Real SSD uses feature maps at 6 different resolutions (38x38, 19x19, 10x10, 5x5, 3x3, 1x1). Smaller objects are detected on higher-resolution maps.
- **Focal Loss**: Addresses the extreme class imbalance (most anchors are background) by down-weighting easy negatives. Key innovation of RetinaNet.
- **YOLO vs SSD vs Faster R-CNN**: YOLO divides image into grid cells (one prediction per cell); SSD uses multi-scale anchors; Faster R-CNN has a separate Region Proposal Network. Each trades speed for accuracy differently.
- **Anchor-free detectors**: Recent work (FCOS, CenterNet) eliminates anchors entirely, predicting center points and distances to box edges. Simpler but requires careful feature map assignment.
- **mAP metric**: Mean Average Precision across IoU thresholds — the standard evaluation metric for detection benchmarks (COCO, PASCAL VOC).
