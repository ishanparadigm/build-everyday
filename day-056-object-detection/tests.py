"""
Day 056: Object Detection Tests

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
import numpy as np
from my_solution import (
    corner_to_center,
    center_to_corner,
    compute_iou,
    generate_anchors,
    match_anchors_to_gt,
    encode_offsets,
    decode_offsets,
    nms,
    multiclass_nms,
)


class TestBoxConversion(unittest.TestCase):
    """Test bounding box format conversions."""

    def test_corner_to_center(self):
        boxes = np.array([[0, 0, 10, 10], [20, 30, 60, 90]], dtype=np.float64)
        result = corner_to_center(boxes)
        expected = np.array([[5, 5, 10, 10], [40, 60, 40, 60]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_center_to_corner(self):
        boxes = np.array([[5, 5, 10, 10], [40, 60, 40, 60]], dtype=np.float64)
        result = center_to_corner(boxes)
        expected = np.array([[0, 0, 10, 10], [20, 30, 60, 90]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_round_trip(self):
        original = np.array([[3, 7, 15, 22], [0, 0, 100, 50]], dtype=np.float64)
        result = center_to_corner(corner_to_center(original))
        np.testing.assert_array_almost_equal(result, original)


class TestIoU(unittest.TestCase):
    """Test Intersection over Union computation."""

    def test_identical_boxes(self):
        box = np.array([[10, 10, 50, 50]], dtype=np.float64)
        iou = compute_iou(box, box)
        self.assertAlmostEqual(iou[0, 0], 1.0, places=5)

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        iou = compute_iou(a, b)
        self.assertAlmostEqual(iou[0, 0], 0.0, places=5)

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        iou = compute_iou(a, b)
        # Intersection: 5x5=25, Union: 100+100-25=175
        self.assertAlmostEqual(iou[0, 0], 25.0 / 175.0, places=5)

    def test_contained_box(self):
        outer = np.array([[0, 0, 100, 100]], dtype=np.float64)
        inner = np.array([[25, 25, 75, 75]], dtype=np.float64)
        iou = compute_iou(outer, inner)
        # Intersection: 50*50=2500, Union: 10000+2500-2500=10000
        self.assertAlmostEqual(iou[0, 0], 2500.0 / 10000.0, places=5)

    def test_pairwise_shape(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15], [0, 0, 5, 5], [25, 25, 35, 35]], dtype=np.float64)
        iou = compute_iou(a, b)
        self.assertEqual(iou.shape, (2, 3))


class TestAnchors(unittest.TestCase):
    """Test anchor box generation."""

    def test_anchor_count(self):
        anchors = generate_anchors(256, 8, [1.0], [1.0])
        self.assertEqual(len(anchors), 64)  # 8*8*1*1

    def test_anchor_count_multi(self):
        anchors = generate_anchors(256, 4, [0.5, 1.0], [1.0, 2.0])
        self.assertEqual(len(anchors), 4 * 4 * 2 * 2)  # 64

    def test_anchors_within_image(self):
        anchors = generate_anchors(256, 8, [0.5, 1.0], [0.5, 1.0, 2.0])
        self.assertTrue(np.all(anchors[:, 0] >= 0))
        self.assertTrue(np.all(anchors[:, 1] >= 0))
        self.assertTrue(np.all(anchors[:, 2] <= 256))
        self.assertTrue(np.all(anchors[:, 3] <= 256))

    def test_anchor_shape(self):
        anchors = generate_anchors(256, 8, [1.0], [1.0])
        self.assertEqual(anchors.shape, (64, 4))


class TestOffsets(unittest.TestCase):
    """Test offset encoding and decoding."""

    def test_encode_decode_round_trip(self):
        anchors = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float64)
        gt_boxes = np.array([[15, 12, 55, 48], [90, 110, 210, 190]], dtype=np.float64)
        offsets = encode_offsets(anchors, gt_boxes)
        decoded = decode_offsets(anchors, offsets)
        np.testing.assert_array_almost_equal(decoded, gt_boxes, decimal=4)

    def test_zero_offset_for_identical(self):
        boxes = np.array([[20, 30, 80, 90]], dtype=np.float64)
        offsets = encode_offsets(boxes, boxes)
        np.testing.assert_array_almost_equal(offsets, np.zeros((1, 4)), decimal=5)


class TestNMS(unittest.TestCase):
    """Test non-maximum suppression."""

    def test_no_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float64)
        scores = np.array([0.9, 0.8])
        kept = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)

    def test_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
        scores = np.array([0.9, 0.8])
        kept = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], 0)  # Higher score survives

    def test_partial_suppression(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],
            [100, 100, 150, 150],
        ], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.7])
        kept = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)
        self.assertIn(0, kept)
        self.assertIn(2, kept)

    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float64)
        scores = np.array([])
        kept = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(len(kept), 0)

    def test_keeps_highest_score(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float64)
        scores = np.array([0.5, 0.9])
        kept = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(kept[0], 1)  # Higher score first


class TestMulticlassNMS(unittest.TestCase):
    """Test per-class NMS."""

    def test_basic(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
        ], dtype=np.float64)
        scores = np.array([
            [0.9, 0.1],
            [0.8, 0.1],
            [0.1, 0.9],
        ])
        kept_boxes, kept_labels, kept_scores = multiclass_nms(
            boxes, scores, score_threshold=0.3, iou_threshold=0.5
        )
        # Class 1: boxes 0 and 1 overlap, keep 0. Class 2: box 2.
        self.assertEqual(len(kept_boxes), 2)


class TestAnchorMatching(unittest.TestCase):
    """Test anchor-to-ground-truth matching."""

    def test_basic_matching(self):
        anchors = generate_anchors(256, 4, [1.0], [1.0])
        gt_boxes = np.array([[20, 20, 80, 80]], dtype=np.float64)
        gt_labels = np.array([1])
        labels, gt_idx = match_anchors_to_gt(anchors, gt_boxes, gt_labels)
        # At least one anchor must be positive
        self.assertTrue((labels > 0).any())
        # All labels must be in {-1, 0, 1}
        self.assertTrue(np.all(np.isin(labels, [-1, 0, 1])))

    def test_no_gt_boxes(self):
        anchors = generate_anchors(256, 4, [1.0], [1.0])
        gt_boxes = np.zeros((0, 4), dtype=np.float64)
        gt_labels = np.zeros(0, dtype=np.int64)
        labels, gt_idx = match_anchors_to_gt(anchors, gt_boxes, gt_labels)
        self.assertTrue(np.all(labels == 0))


if __name__ == "__main__":
    unittest.main()
