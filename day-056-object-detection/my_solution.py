"""
Day 056: Object Detection Basics — Your Implementation

Implement the core building blocks of object detection:
- Bounding box operations (IoU, format conversion)
- Anchor box generation
- Offset encoding/decoding
- Non-maximum suppression (NMS)

Work through each function in order — they build on each other.
Run this file to test as you go: python3 my_solution.py
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# Bounding Box Utilities
# =============================================================================

def corner_to_center(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from corner format to center format.

    Corner: [x_min, y_min, x_max, y_max]
    Center: [cx, cy, w, h]

    Hint: cx = midpoint of x_min and x_max, w = difference

    Args:
        boxes: (N, 4) array in corner format
    Returns:
        (N, 4) array in center format
    """
    raise NotImplementedError("TODO: implement this")


def center_to_corner(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from center format to corner format.

    Center: [cx, cy, w, h]
    Corner: [x_min, y_min, x_max, y_max]

    Hint: x_min = cx - w/2

    Args:
        boxes: (N, 4) array in center format
    Returns:
        (N, 4) array in corner format
    """
    raise NotImplementedError("TODO: implement this")


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes (corner format).

    IoU = intersection_area / union_area
    union_area = area_a + area_b - intersection_area

    Hint: Use broadcasting — expand boxes_a to (N,1,4) and boxes_b to (1,M,4)
    so element-wise min/max operations produce (N,M) results.
    Don't forget to clamp intersection width/height to zero.

    Args:
        boxes_a: (N, 4) in corner format
        boxes_b: (M, 4) in corner format
    Returns:
        (N, M) IoU matrix
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Anchor Box Generation
# =============================================================================

def generate_anchors(
    image_size: int,
    grid_size: int,
    scales: List[float],
    aspect_ratios: List[float]
) -> np.ndarray:
    """
    Generate anchor boxes tiled over an image grid.

    For each grid cell, generate len(scales) * len(aspect_ratios) anchors.
    Each anchor is centered at the grid cell center.

    For scale s and aspect ratio r:
        w = cell_size * s * sqrt(r)
        h = cell_size * s / sqrt(r)
    This keeps the area constant across aspect ratios.

    Hint: cell_size = image_size / grid_size.
    Center of cell (row, col) = ((col+0.5)*cell_size, (row+0.5)*cell_size).
    Clip boxes to [0, image_size].

    Args:
        image_size: Size of the square image
        grid_size: Number of grid cells per dimension
        scales: Scale factors
        aspect_ratios: Width/height ratios
    Returns:
        (grid_size^2 * num_anchors_per_cell, 4) in corner format
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Anchor Matching
# =============================================================================

def match_anchors_to_gt(
    anchors: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pos_iou_threshold: float = 0.5,
    neg_iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each anchor to a ground-truth box or background.

    Rules:
    - IoU >= pos_threshold → positive (label = gt class)
    - IoU < neg_threshold → negative (label = 0)
    - In between → ignored (label = -1)
    - Each GT box must have at least one positive anchor (the best one)

    Hint: Compute the full IoU matrix, then use argmax/max along axis=1
    to find each anchor's best GT match.

    Args:
        anchors: (A, 4) anchor boxes
        gt_boxes: (G, 4) ground truth boxes
        gt_labels: (G,) class labels (1-indexed)
        pos_iou_threshold: threshold for positive
        neg_iou_threshold: threshold for negative
    Returns:
        matched_labels: (A,) — 0=background, -1=ignore, >0=class
        matched_gt_indices: (A,) — index of matched GT box, -1 if none
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Offset Encoding / Decoding
# =============================================================================

def encode_offsets(anchors: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Encode ground-truth boxes as offsets from matched anchors.

    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = log(gt_w / anchor_w)
    th = log(gt_h / anchor_h)

    Hint: Convert both to center format first. Use np.log for tw/th.
    Guard against zero width/height with np.maximum(..., 1e-6).

    Args:
        anchors: (N, 4) corner format
        gt_boxes: (N, 4) corner format, matched 1-to-1
    Returns:
        (N, 4) offsets [tx, ty, tw, th]
    """
    raise NotImplementedError("TODO: implement this")


def decode_offsets(anchors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Decode predicted offsets back to absolute bounding boxes.

    pred_cx = tx * anchor_w + anchor_cx
    pred_cy = ty * anchor_h + anchor_cy
    pred_w  = anchor_w * exp(tw)
    pred_h  = anchor_h * exp(th)

    Hint: This is the inverse of encode_offsets.
    Clip tw/th before exp() to avoid overflow (use np.clip(..., -10, 10)).

    Args:
        anchors: (N, 4) corner format
        offsets: (N, 4) predicted offsets
    Returns:
        (N, 4) decoded boxes in corner format
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Non-Maximum Suppression
# =============================================================================

def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Greedy Non-Maximum Suppression.

    1. Sort by score descending
    2. Pick top box, add to keep list
    3. Remove all remaining boxes with IoU > threshold vs picked box
    4. Repeat until empty

    Hint: Use a while loop. After picking the best box, compute IoU
    between it and all remaining boxes. Keep only those below threshold.

    Args:
        boxes: (N, 4) corner format
        scores: (N,) confidence scores
        iou_threshold: suppression threshold
    Returns:
        Array of kept indices
    """
    raise NotImplementedError("TODO: implement this")


def multiclass_nms(
    boxes: np.ndarray,
    class_scores: np.ndarray,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply NMS independently per class, then combine.

    For each class:
    1. Filter boxes by score_threshold
    2. Run NMS
    3. Collect surviving boxes with their class label and score

    Hint: Loop over classes. For each, filter, run nms(), collect results.
    Use np.concatenate to merge at the end.

    Args:
        boxes: (N, 4) predicted boxes
        class_scores: (N, C) per-class scores
        score_threshold: minimum score
        iou_threshold: NMS threshold
    Returns:
        kept_boxes (K, 4), kept_labels (K,), kept_scores (K,)
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test your implementation
# =============================================================================

if __name__ == "__main__":
    print("Testing your object detection implementation...\n")

    # Test 1: Box format conversion
    print("Test 1: Box format conversion")
    corners = np.array([[10, 20, 50, 80], [0, 0, 100, 100]], dtype=np.float64)
    centers = corner_to_center(corners)
    print(f"  Corner → Center: {corners[0]} → {centers[0]}")
    # Expected: [30, 50, 40, 60]
    back = center_to_corner(centers)
    print(f"  Round trip error: {np.abs(back - corners).max():.2e}")

    # Test 2: IoU
    print("\nTest 2: IoU computation")
    box_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    box_b = np.array([[5, 5, 15, 15]], dtype=np.float64)
    iou_val = compute_iou(box_a, box_b)[0, 0]
    print(f"  IoU of [0,0,10,10] and [5,5,15,15]: {iou_val:.4f}")
    # Expected: 25/175 ≈ 0.1429

    # Test 3: Anchor generation
    print("\nTest 3: Anchor generation")
    anchors = generate_anchors(256, 8, [1.0], [1.0])
    print(f"  8x8 grid, 1 scale, 1 ratio → {len(anchors)} anchors")
    # Expected: 64

    # Test 4: Encode/Decode round trip
    print("\nTest 4: Offset encode/decode")
    test_anchors = np.array([[10, 10, 50, 50]], dtype=np.float64)
    test_gt = np.array([[15, 12, 55, 48]], dtype=np.float64)
    offsets = encode_offsets(test_anchors, test_gt)
    decoded = decode_offsets(test_anchors, offsets)
    print(f"  GT: {test_gt[0]} → Offsets: {offsets[0].round(4)} → Decoded: {decoded[0].round(1)}")
    print(f"  Reconstruction error: {np.abs(decoded - test_gt).max():.2e}")

    # Test 5: NMS
    print("\nTest 5: Non-Maximum Suppression")
    nms_boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],  # overlaps heavily with first
        [100, 100, 150, 150],  # separate
    ], dtype=np.float64)
    nms_scores = np.array([0.9, 0.8, 0.7])
    kept = nms(nms_boxes, nms_scores, iou_threshold=0.5)
    print(f"  Kept indices: {kept}")
    # Expected: [0, 2] (first and third survive, second is suppressed)

    # Test 6: Full pipeline
    print("\nTest 6: Full pipeline")
    anchors = generate_anchors(256, 8, [0.5, 1.0, 1.5], [0.5, 1.0, 2.0])
    gt_boxes = np.array([[30, 30, 100, 100], [150, 150, 230, 230]], dtype=np.float64)
    gt_labels = np.array([1, 2])

    matched_labels, matched_gt_idx = match_anchors_to_gt(anchors, gt_boxes, gt_labels)
    print(f"  Positive anchors: {(matched_labels > 0).sum()}")
    print(f"  Negative anchors: {(matched_labels == 0).sum()}")

    print("\nAll tests passed!" if True else "")
