"""
Day 056: Object Detection Basics — Single-Shot Detector Pipeline

Implements the core building blocks of object detection from scratch:
- Bounding box operations (IoU, format conversion)
- Anchor box generation with multiple scales and aspect ratios
- Offset encoding/decoding for bounding box regression
- Non-maximum suppression (NMS)
- End-to-end simulated detection pipeline

All implemented with NumPy only — no deep learning frameworks.
"""

import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# Bounding Box Utilities
# =============================================================================

def corner_to_center(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from corner format to center format.

    Corner format: [x_min, y_min, x_max, y_max]
    Center format: [cx, cy, w, h]

    Why two formats? Corner format is natural for IoU computation (intersection
    requires min/max operations). Center format is natural for anchor generation
    and offset encoding (deltas are relative to center and size).

    Args:
        boxes: (N, 4) array in corner format
    Returns:
        (N, 4) array in center format
    """
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return np.stack([cx, cy, w, h], axis=1)


def center_to_corner(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from center format to corner format.

    Args:
        boxes: (N, 4) array in center format [cx, cy, w, h]
    Returns:
        (N, 4) array in corner format [x_min, y_min, x_max, y_max]
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = cx - w / 2.0
    y_min = cy - h / 2.0
    x_max = cx + w / 2.0
    y_max = cy + h / 2.0
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes in corner format.

    This is the workhorse function of object detection. It's called during:
    1. Anchor matching (which anchors are responsible for which ground-truth boxes)
    2. NMS (which detections overlap and should be suppressed)
    3. Evaluation (do predicted boxes match ground-truth boxes)

    We compute it in a vectorized way using broadcasting:
    - boxes_a has shape (N, 4), boxes_b has shape (M, 4)
    - We expand to (N, 1, 4) and (1, M, 4) so element-wise operations give (N, M)

    Args:
        boxes_a: (N, 4) array in corner format
        boxes_b: (M, 4) array in corner format
    Returns:
        (N, M) IoU matrix where result[i, j] = IoU(boxes_a[i], boxes_b[j])
    """
    # Expand dims for broadcasting: (N, 1, 4) vs (1, M, 4) -> (N, M) results
    a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, M, 4)

    # Intersection rectangle: take the tighter bounds
    inter_x_min = np.maximum(a[..., 0], b[..., 0])
    inter_y_min = np.maximum(a[..., 1], b[..., 1])
    inter_x_max = np.minimum(a[..., 2], b[..., 2])
    inter_y_max = np.minimum(a[..., 3], b[..., 3])

    # Clamp to zero: if boxes don't overlap, intersection width/height is negative
    inter_w = np.maximum(0.0, inter_x_max - inter_x_min)
    inter_h = np.maximum(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Areas of each box
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)

    # Union = A + B - intersection (inclusion-exclusion principle)
    union_area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area

    # Avoid division by zero for degenerate boxes
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


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

    The core idea: instead of predicting bounding boxes from scratch, we
    pre-define a set of reference boxes (anchors) at each spatial location.
    The model then predicts small adjustments to these anchors.

    At each grid cell, we generate len(scales) * len(aspect_ratios) anchors.

    For a given base_size, scale s, and aspect ratio r:
        w = base_size * s * sqrt(r)
        h = base_size * s / sqrt(r)

    Why sqrt(r)? We want the anchor area to be approximately (base_size * s)^2
    regardless of aspect ratio. If w = base * s * sqrt(r) and h = base * s / sqrt(r),
    then w * h = (base * s)^2, which is constant across ratios.

    Args:
        image_size: Size of the (square) image in pixels
        grid_size: Number of grid cells along each dimension
        scales: List of scale factors (e.g., [0.5, 1.0, 1.5])
        aspect_ratios: List of width/height ratios (e.g., [0.5, 1.0, 2.0])
    Returns:
        (grid_size * grid_size * num_anchors_per_cell, 4) array in corner format
    """
    cell_size = image_size / grid_size
    anchors = []

    for row in range(grid_size):
        for col in range(grid_size):
            # Center of this grid cell in pixel coordinates
            cx = (col + 0.5) * cell_size
            cy = (row + 0.5) * cell_size

            for scale in scales:
                for ratio in aspect_ratios:
                    # Compute anchor width and height
                    # Area = (cell_size * scale)^2 regardless of ratio
                    w = cell_size * scale * np.sqrt(ratio)
                    h = cell_size * scale / np.sqrt(ratio)

                    # Convert to corner format and clip to image bounds
                    x_min = max(0, cx - w / 2)
                    y_min = max(0, cy - h / 2)
                    x_max = min(image_size, cx + w / 2)
                    y_max = min(image_size, cy + h / 2)

                    anchors.append([x_min, y_min, x_max, y_max])

    return np.array(anchors, dtype=np.float64)


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
    Assign each anchor to a ground-truth box (or background).

    The matching strategy:
    1. Compute IoU between all anchors and all ground-truth boxes
    2. For each anchor, find the ground-truth box with highest IoU
    3. If max IoU >= pos_threshold: anchor is POSITIVE (label = gt class)
    4. If max IoU < neg_threshold: anchor is NEGATIVE (label = 0 = background)
    5. Otherwise: anchor is IGNORED (label = -1, not used in loss)

    Additionally, for each ground-truth box, we ensure at least one anchor is
    assigned to it (the one with highest IoU), even if it's below the threshold.
    This prevents rare objects from having zero positive anchors.

    Args:
        anchors: (A, 4) anchor boxes in corner format
        gt_boxes: (G, 4) ground-truth boxes in corner format
        gt_labels: (G,) integer class labels for each ground-truth box
        pos_iou_threshold: IoU threshold for positive matching
        neg_iou_threshold: IoU threshold below which anchor is negative
    Returns:
        matched_labels: (A,) class label for each anchor (0=bg, -1=ignore, >0=class)
        matched_gt_indices: (A,) index of matched ground-truth box (-1 if none)
    """
    num_anchors = anchors.shape[0]

    # Handle edge case: no ground-truth boxes in this image
    if gt_boxes.shape[0] == 0:
        return np.zeros(num_anchors, dtype=np.int64), np.full(num_anchors, -1, dtype=np.int64)

    # (A, G) IoU matrix
    iou_matrix = compute_iou(anchors, gt_boxes)

    # For each anchor, find best matching ground-truth box
    best_gt_iou = iou_matrix.max(axis=1)       # (A,)
    best_gt_idx = iou_matrix.argmax(axis=1)     # (A,)

    # Initialize all as ignored (-1)
    matched_labels = np.full(num_anchors, -1, dtype=np.int64)
    matched_gt_indices = np.full(num_anchors, -1, dtype=np.int64)

    # Negative: IoU below negative threshold
    matched_labels[best_gt_iou < neg_iou_threshold] = 0

    # Positive: IoU above positive threshold
    pos_mask = best_gt_iou >= pos_iou_threshold
    matched_labels[pos_mask] = gt_labels[best_gt_idx[pos_mask]]
    matched_gt_indices[pos_mask] = best_gt_idx[pos_mask]

    # Ensure every ground-truth box has at least one positive anchor
    # For each GT box, the anchor with highest IoU is forced positive
    best_anchor_per_gt = iou_matrix.argmax(axis=0)  # (G,)
    for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
        matched_labels[anchor_idx] = gt_labels[gt_idx]
        matched_gt_indices[anchor_idx] = gt_idx

    return matched_labels, matched_gt_indices


# =============================================================================
# Bounding Box Offset Encoding / Decoding
# =============================================================================

def encode_offsets(anchors: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Encode ground-truth boxes as offsets from anchors.

    This is the regression target for training. Instead of predicting absolute
    box coordinates (which vary wildly), we predict deltas relative to anchors:

        tx = (gt_cx - anchor_cx) / anchor_w    (translation, normalized by size)
        ty = (gt_cy - anchor_cy) / anchor_h
        tw = log(gt_w / anchor_w)               (log-scale for width)
        th = log(gt_h / anchor_h)               (log-scale for height)

    Why log for width/height?
    - Width must be positive. If the model predicts tw, then gt_w = anchor_w * exp(tw),
      which is always positive regardless of tw's sign.
    - The log transform makes the prediction symmetric: doubling and halving width
      correspond to tw = +0.693 and tw = -0.693 (equal magnitude).

    Why normalize translation by anchor size?
    - A 5px shift means very different things for a 20px anchor vs a 200px anchor.
      Normalizing makes the targets scale-invariant.

    Args:
        anchors: (N, 4) in corner format
        gt_boxes: (N, 4) in corner format (matched 1-to-1 with anchors)
    Returns:
        (N, 4) encoded offsets [tx, ty, tw, th]
    """
    anchors_center = corner_to_center(anchors)
    gt_center = corner_to_center(gt_boxes)

    # Prevent division by zero / log of zero
    anchor_w = np.maximum(anchors_center[:, 2], 1e-6)
    anchor_h = np.maximum(anchors_center[:, 3], 1e-6)
    gt_w = np.maximum(gt_center[:, 2], 1e-6)
    gt_h = np.maximum(gt_center[:, 3], 1e-6)

    tx = (gt_center[:, 0] - anchors_center[:, 0]) / anchor_w
    ty = (gt_center[:, 1] - anchors_center[:, 1]) / anchor_h
    tw = np.log(gt_w / anchor_w)
    th = np.log(gt_h / anchor_h)

    return np.stack([tx, ty, tw, th], axis=1)


def decode_offsets(anchors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Decode predicted offsets back to absolute bounding boxes.

    Inverse of encode_offsets:
        pred_cx = tx * anchor_w + anchor_cx
        pred_cy = ty * anchor_h + anchor_cy
        pred_w  = anchor_w * exp(tw)
        pred_h  = anchor_h * exp(th)

    Args:
        anchors: (N, 4) in corner format
        offsets: (N, 4) predicted offsets [tx, ty, tw, th]
    Returns:
        (N, 4) decoded boxes in corner format
    """
    anchors_center = corner_to_center(anchors)

    anchor_w = anchors_center[:, 2]
    anchor_h = anchors_center[:, 3]

    pred_cx = offsets[:, 0] * anchor_w + anchors_center[:, 0]
    pred_cy = offsets[:, 1] * anchor_h + anchors_center[:, 1]
    # Clamp tw/th to prevent exp() overflow
    pred_w = anchor_w * np.exp(np.clip(offsets[:, 2], -10, 10))
    pred_h = anchor_h * np.exp(np.clip(offsets[:, 3], -10, 10))

    decoded_center = np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)
    return center_to_corner(decoded_center)


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

    Why NMS is necessary: in a typical detection output, many overlapping anchors
    will fire for the same object. Without NMS, you'd report 10-50 boxes per object.
    NMS keeps only the best (highest confidence) detection and removes all others
    that significantly overlap with it.

    Algorithm:
    1. Sort detections by score (highest first)
    2. Pick the top detection, add to output
    3. Compute IoU between this detection and all remaining
    4. Remove any detection with IoU > threshold (they're likely duplicates)
    5. Repeat with the next highest-scoring remaining detection

    Time complexity: O(N^2) in the worst case (all boxes survive), but typically
    much faster because most boxes get suppressed early.

    Args:
        boxes: (N, 4) in corner format
        scores: (N,) confidence scores
        iou_threshold: suppress boxes with IoU > this value
    Returns:
        Array of indices of kept detections, sorted by score
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # Sort by score descending
    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        # Pick the highest-scoring remaining detection
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        # Compute IoU between current box and all remaining boxes
        remaining = order[1:]
        current_box = boxes[current:current+1]  # (1, 4)
        remaining_boxes = boxes[remaining]       # (R, 4)

        ious = compute_iou(current_box, remaining_boxes)[0]  # (R,)

        # Keep only boxes with IoU below threshold (i.e., not duplicates)
        mask = ious <= iou_threshold
        order = remaining[mask]

    return np.array(keep, dtype=np.int64)


def multiclass_nms(
    boxes: np.ndarray,
    class_scores: np.ndarray,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply NMS independently per class, then combine results.

    Per-class NMS is standard because two objects of different classes at the
    same location (e.g., a person riding a horse) should both be detected,
    not suppressed.

    Args:
        boxes: (N, 4) predicted boxes in corner format
        class_scores: (N, C) per-class confidence scores (C classes, excluding bg)
        score_threshold: minimum score to consider a detection
        iou_threshold: NMS IoU threshold
    Returns:
        kept_boxes: (K, 4) final detected boxes
        kept_labels: (K,) class labels (1-indexed)
        kept_scores: (K,) confidence scores
    """
    num_classes = class_scores.shape[1]
    all_boxes = []
    all_labels = []
    all_scores = []

    for cls_idx in range(num_classes):
        cls_scores = class_scores[:, cls_idx]

        # Filter by score threshold
        mask = cls_scores > score_threshold
        if not mask.any():
            continue

        cls_boxes = boxes[mask]
        cls_conf = cls_scores[mask]

        # Apply NMS for this class
        keep_idx = nms(cls_boxes, cls_conf, iou_threshold)

        all_boxes.append(cls_boxes[keep_idx])
        all_labels.append(np.full(len(keep_idx), cls_idx + 1, dtype=np.int64))
        all_scores.append(cls_conf[keep_idx])

    if len(all_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0)

    return (
        np.concatenate(all_boxes),
        np.concatenate(all_labels),
        np.concatenate(all_scores)
    )


# =============================================================================
# Simulated Detection Pipeline
# =============================================================================

def create_synthetic_scene(
    image_size: int = 256,
    num_objects: int = 4,
    num_classes: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic scene with random ground-truth bounding boxes.

    Creates non-overlapping objects of varying sizes to simulate a typical
    detection scenario (e.g., objects in a warehouse or cars on a road).

    Args:
        image_size: size of the square image
        num_objects: number of objects to generate
        num_classes: number of object classes
        seed: random seed for reproducibility
    Returns:
        gt_boxes: (num_objects, 4) in corner format
        gt_labels: (num_objects,) class labels (1-indexed)
    """
    rng = np.random.RandomState(seed)

    gt_boxes = []
    gt_labels = []

    for _ in range(num_objects):
        # Random box size between 20% and 40% of image
        w = rng.uniform(0.15, 0.35) * image_size
        h = rng.uniform(0.15, 0.35) * image_size

        # Random position ensuring box stays within image
        x_min = rng.uniform(0, image_size - w)
        y_min = rng.uniform(0, image_size - h)

        gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
        gt_labels.append(rng.randint(1, num_classes + 1))

    return np.array(gt_boxes), np.array(gt_labels)


def simulate_detection(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    anchors: np.ndarray,
    num_classes: int = 3,
    noise_std: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a neural network's detection output.

    For positive anchors (matched to a GT box), we generate:
    - Box offsets = true offsets + small noise (simulating imperfect regression)
    - High class score for the correct class, low for others

    For negative anchors, we generate random low-confidence predictions.

    This lets us test the full decode + NMS pipeline without training a real model.

    Args:
        gt_boxes: (G, 4) ground truth boxes
        gt_labels: (G,) ground truth labels (1-indexed)
        anchors: (A, 4) anchor boxes
        num_classes: number of object classes
        noise_std: standard deviation of offset noise
        seed: random seed
    Returns:
        pred_offsets: (A, 4) predicted box offsets
        pred_scores: (A, num_classes) per-class confidence scores
    """
    rng = np.random.RandomState(seed)
    num_anchors = anchors.shape[0]

    # Match anchors to ground truth
    matched_labels, matched_gt_idx = match_anchors_to_gt(anchors, gt_boxes, gt_labels)

    # Initialize predictions with random noise (background predictions)
    pred_offsets = rng.randn(num_anchors, 4) * 0.3
    pred_scores = rng.uniform(0.0, 0.15, size=(num_anchors, num_classes))

    # For positive anchors, generate realistic predictions
    pos_mask = matched_labels > 0
    pos_indices = np.where(pos_mask)[0]

    if len(pos_indices) > 0:
        # True offsets + noise
        pos_gt_boxes = gt_boxes[matched_gt_idx[pos_indices]]
        pos_anchors = anchors[pos_indices]
        true_offsets = encode_offsets(pos_anchors, pos_gt_boxes)
        pred_offsets[pos_indices] = true_offsets + rng.randn(len(pos_indices), 4) * noise_std

        # High score for correct class, low for others
        for i, anchor_idx in enumerate(pos_indices):
            cls = matched_labels[anchor_idx] - 1  # Convert to 0-indexed
            pred_scores[anchor_idx] = rng.uniform(0.0, 0.1, size=num_classes)
            pred_scores[anchor_idx, cls] = rng.uniform(0.7, 0.95)

    return pred_offsets, pred_scores


def run_detection_pipeline(
    anchors: np.ndarray,
    pred_offsets: np.ndarray,
    pred_scores: np.ndarray,
    image_size: int = 256,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full detection inference pipeline.

    Steps:
    1. Decode predicted offsets to get absolute bounding boxes
    2. Clip boxes to image boundaries
    3. Apply multi-class NMS

    Args:
        anchors: (A, 4) anchor boxes
        pred_offsets: (A, 4) predicted offsets
        pred_scores: (A, C) per-class confidence scores
        image_size: image dimension for clipping
        score_threshold: minimum confidence to keep
        nms_iou_threshold: IoU threshold for NMS
    Returns:
        final_boxes, final_labels, final_scores
    """
    # Step 1: Decode offsets to absolute coordinates
    decoded_boxes = decode_offsets(anchors, pred_offsets)

    # Step 2: Clip to image bounds
    decoded_boxes[:, 0] = np.clip(decoded_boxes[:, 0], 0, image_size)
    decoded_boxes[:, 1] = np.clip(decoded_boxes[:, 1], 0, image_size)
    decoded_boxes[:, 2] = np.clip(decoded_boxes[:, 2], 0, image_size)
    decoded_boxes[:, 3] = np.clip(decoded_boxes[:, 3], 0, image_size)

    # Step 3: Multi-class NMS
    final_boxes, final_labels, final_scores = multiclass_nms(
        decoded_boxes, pred_scores, score_threshold, nms_iou_threshold
    )

    return final_boxes, final_labels, final_scores


# =============================================================================
# Main: Demonstrate the full pipeline
# =============================================================================

if __name__ == "__main__":
    IMAGE_SIZE = 256
    GRID_SIZE = 8
    NUM_CLASSES = 3
    CLASS_NAMES = {1: "car", 2: "person", 3: "dog"}

    print("=" * 70)
    print("OBJECT DETECTION BASICS — FULL PIPELINE DEMO")
    print("=" * 70)

    # ---- Step 1: Generate Anchor Boxes ----
    print("\n--- Step 1: Anchor Box Generation ---")
    scales = [0.5, 1.0, 1.5]
    aspect_ratios = [0.5, 1.0, 2.0]
    anchors = generate_anchors(IMAGE_SIZE, GRID_SIZE, scales, aspect_ratios)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Scales: {scales}, Aspect ratios: {aspect_ratios}")
    print(f"Anchors per cell: {len(scales) * len(aspect_ratios)}")
    print(f"Total anchors: {len(anchors)}")
    print(f"Sample anchor (first): {anchors[0].round(1)}")
    print(f"Sample anchor (center cell): {anchors[len(anchors)//2].round(1)}")

    # ---- Step 2: Create Synthetic Scene ----
    print("\n--- Step 2: Synthetic Ground Truth ---")
    gt_boxes, gt_labels = create_synthetic_scene(IMAGE_SIZE, num_objects=4, num_classes=NUM_CLASSES)
    print(f"Ground truth objects: {len(gt_boxes)}")
    for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        print(f"  Object {i}: class={CLASS_NAMES[label]} ({label}), "
              f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    # ---- Step 3: IoU Demonstration ----
    print("\n--- Step 3: IoU Computation ---")
    # Show IoU between ground-truth boxes
    gt_iou = compute_iou(gt_boxes, gt_boxes)
    print("IoU matrix between GT boxes (diagonal = 1.0, self-IoU):")
    for i in range(len(gt_boxes)):
        row = " ".join(f"{v:.3f}" for v in gt_iou[i])
        print(f"  GT {i}: [{row}]")

    # ---- Step 4: Anchor Matching ----
    print("\n--- Step 4: Anchor-to-GT Matching ---")
    matched_labels, matched_gt_idx = match_anchors_to_gt(anchors, gt_boxes, gt_labels)
    num_positive = (matched_labels > 0).sum()
    num_negative = (matched_labels == 0).sum()
    num_ignored = (matched_labels == -1).sum()
    print(f"Positive anchors (IoU >= 0.5): {num_positive}")
    print(f"Negative anchors (IoU < 0.3):  {num_negative}")
    print(f"Ignored anchors (in between):  {num_ignored}")
    print(f"Positive ratio: {num_positive / len(anchors):.1%}")
    print("  (Typical: 1-5% of anchors are positive — extreme class imbalance!)")

    # ---- Step 5: Offset Encoding/Decoding Round Trip ----
    print("\n--- Step 5: Offset Encoding/Decoding ---")
    pos_mask = matched_labels > 0
    pos_anchors = anchors[pos_mask][:5]  # First 5 positive anchors
    pos_gt = gt_boxes[matched_gt_idx[pos_mask]][:5]

    offsets = encode_offsets(pos_anchors, pos_gt)
    decoded = decode_offsets(pos_anchors, offsets)

    print("Encode → Decode round trip (should perfectly reconstruct GT):")
    for i in range(min(3, len(pos_anchors))):
        print(f"  Anchor:  [{pos_anchors[i][0]:.1f}, {pos_anchors[i][1]:.1f}, "
              f"{pos_anchors[i][2]:.1f}, {pos_anchors[i][3]:.1f}]")
        print(f"  GT:      [{pos_gt[i][0]:.1f}, {pos_gt[i][1]:.1f}, "
              f"{pos_gt[i][2]:.1f}, {pos_gt[i][3]:.1f}]")
        print(f"  Offsets: [{offsets[i][0]:.3f}, {offsets[i][1]:.3f}, "
              f"{offsets[i][2]:.3f}, {offsets[i][3]:.3f}]")
        print(f"  Decoded: [{decoded[i][0]:.1f}, {decoded[i][1]:.1f}, "
              f"{decoded[i][2]:.1f}, {decoded[i][3]:.1f}]")
        max_err = np.abs(decoded[i] - pos_gt[i]).max()
        print(f"  Max reconstruction error: {max_err:.2e}")
        print()

    # ---- Step 6: Simulated Detection ----
    print("--- Step 6: Simulated Detection Output ---")
    pred_offsets, pred_scores = simulate_detection(
        gt_boxes, gt_labels, anchors, NUM_CLASSES, noise_std=0.1
    )
    print(f"Predicted offsets shape: {pred_offsets.shape}")
    print(f"Predicted scores shape: {pred_scores.shape}")
    print(f"Max score per class: {pred_scores.max(axis=0).round(3)}")

    # ---- Step 7: Full Detection Pipeline ----
    print("\n--- Step 7: Full Pipeline (Decode + NMS) ---")
    final_boxes, final_labels, final_scores = run_detection_pipeline(
        anchors, pred_offsets, pred_scores, IMAGE_SIZE,
        score_threshold=0.3, nms_iou_threshold=0.5
    )

    print(f"\nFinal detections after NMS: {len(final_boxes)}")
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        print(f"  Detection {i}: class={CLASS_NAMES[final_labels[i]]} "
              f"(score={final_scores[i]:.3f}), "
              f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    # ---- Step 8: Evaluate Detections Against Ground Truth ----
    print("\n--- Step 8: Detection Quality ---")
    if len(final_boxes) > 0:
        det_gt_iou = compute_iou(final_boxes, gt_boxes)
        print("IoU between detections and GT boxes:")
        for i in range(len(final_boxes)):
            best_gt = det_gt_iou[i].argmax()
            best_iou = det_gt_iou[i, best_gt]
            match = "MATCH" if best_iou > 0.5 else "MISS"
            print(f"  Det {i} ({CLASS_NAMES[final_labels[i]]}) ↔ "
                  f"GT {best_gt} ({CLASS_NAMES[gt_labels[best_gt]]}): "
                  f"IoU={best_iou:.3f} [{match}]")

    # ---- NMS Ablation ----
    print("\n--- Bonus: NMS Ablation ---")
    decoded_all = decode_offsets(anchors, pred_offsets)
    decoded_all[:, 0] = np.clip(decoded_all[:, 0], 0, IMAGE_SIZE)
    decoded_all[:, 1] = np.clip(decoded_all[:, 1], 0, IMAGE_SIZE)
    decoded_all[:, 2] = np.clip(decoded_all[:, 2], 0, IMAGE_SIZE)
    decoded_all[:, 3] = np.clip(decoded_all[:, 3], 0, IMAGE_SIZE)

    # Count detections above score threshold without NMS
    high_score_mask = pred_scores.max(axis=1) > 0.3
    print(f"Detections above score threshold (before NMS): {high_score_mask.sum()}")
    print(f"Detections after NMS: {len(final_boxes)}")
    print(f"Suppression ratio: {1 - len(final_boxes) / max(1, high_score_mask.sum()):.1%}")
    print("  → NMS removes the vast majority of redundant detections!")

    print("\n" + "=" * 70)
    print("Pipeline complete. Key takeaways:")
    print("  1. Anchors define the search space — fewer anchors = faster but might miss objects")
    print("  2. Offset encoding normalizes the regression task across scales")
    print("  3. NMS is critical — without it, each object gets dozens of detections")
    print("  4. The positive/negative ratio is extremely imbalanced (~1-5% positive)")
    print("=" * 70)
