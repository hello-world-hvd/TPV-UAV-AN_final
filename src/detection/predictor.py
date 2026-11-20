"""Lấy box từ checkpoint segmentation YOLOE, bỏ bước dựng mask không cần cho detect."""

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor


class BoxesOnlyPredictor(SegmentationPredictor):
    def construct_results(self, preds, img, orig_imgs, protos=None):
        return DetectionPredictor.construct_results(self, preds, img, orig_imgs)

    def construct_result(self, pred, img, orig_img, img_path, proto=None):
        return DetectionPredictor.construct_result(self, pred, img, orig_img, img_path)
