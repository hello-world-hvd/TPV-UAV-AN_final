"""Kiểm tra NMS cùng nhãn: IoS, IoU, box khác lớp và cấu hình tắt IoS."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection.detector import Detector


class NMSTests(unittest.TestCase):
    def setUp(self):
        self.detector = Detector.__new__(Detector)
        self.detector.merge_iou = .5
        self.detector.merge_ios = .9

    def row(self, box, score=.8, label="xe máy"):
        return dict(bbox_xyxy=box, confidence=score, label=label)

    def test_containment_keeps_higher_confidence(self):
        large = self.row([0, 0, 100, 100], .9)
        small = self.row([10, 10, 30, 30], .8)
        self.assertEqual(self.detector._merge([small, large]), [large])
        small["confidence"] = .95
        self.assertEqual(self.detector._merge([large, small]), [small])

    def test_other_labels_and_adjacent_objects_survive(self):
        rows = [self.row([0, 0, 100, 100]), self.row([10, 10, 30, 30], label="người"),
                self.row([90, 0, 190, 100])]
        self.assertEqual(len(self.detector._merge(rows)), 3)

    def test_disable_ios_preserves_iou_nms(self):
        self.detector.merge_ios = None
        rows = [self.row([0, 0, 100, 100], .9), self.row([10, 10, 30, 30])]
        self.assertEqual(len(self.detector._merge(rows)), 2)
        self.assertEqual(len(self.detector._merge([rows[0], self.row([1, 1, 101, 101])])), 1)


if __name__ == "__main__":
    unittest.main()
