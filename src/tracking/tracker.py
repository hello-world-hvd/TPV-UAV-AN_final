"""Theo dõi từng ID bằng Kalman/Hungarian, bù camera và optical flow; chỉ detection xác nhận track."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.motion import CameraMotion, warp_box


def measurement(box):
    box = np.asarray(box, dtype=float)
    return np.r_[(box[:2] + box[2:]) / 2, np.maximum(1, box[2:] - box[:2])]


def bbox(state):
    half = np.maximum(1, state[2:4]) / 2
    return np.r_[state[:2] - half, state[:2] + half]


def association_costs(tracks, detections, config):
    """Tính đồng thời IoU, khoảng cách tâm và thay đổi kích thước bằng NumPy."""
    costs = np.full((len(tracks), len(detections)), 1e6)
    if not costs.size:
        return costs
    states = np.stack([t["x"][:4] for t in tracks])
    boxes = np.stack([bbox(t["x"]) for t in tracks])
    observed = np.asarray([d["bbox_xyxy"] for d in detections], dtype=float)
    sizes = np.maximum(1, observed[:, 2:] - observed[:, :2])
    centers = (observed[:, :2] + observed[:, 2:]) / 2
    intersection = np.maximum(0, np.minimum(boxes[:, None, 2:], observed[None, :, 2:])
                              - np.maximum(boxes[:, None, :2], observed[None, :, :2])).prod(-1)
    union = (boxes[:, 2:] - boxes[:, :2]).prod(-1)[:, None] + (observed[:, 2:] - observed[:, :2]).prod(-1)[None, :] - intersection
    overlap = intersection / np.maximum(union, 1e-6)
    distance = np.linalg.norm(centers[None, :, :] - states[:, None, :2], axis=-1) / np.maximum(np.linalg.norm(states[:, 2:4], axis=-1), 1)[:, None]
    size_change = np.abs(np.log(sizes[None, :, :] / np.maximum(states[:, None, 2:4], 1))).max(-1)
    same_label = np.asarray([t["label"] for t in tracks])[:, None] == np.asarray([d["label"] for d in detections])[None, :]
    valid = same_label & (size_change < .8) & ((overlap >= config["min_iou"]) | (distance <= config["max_center_distance"]))
    costs[valid] = (1 - overlap + .2 * distance)[valid]
    return costs


class Tracker:
    def __init__(self, config):
        self.config = config
        self.motion = CameraMotion(config["motion_max_side"], config["flow_refresh_interval"])
        self.tracks = []
        self.next_id = 1
        self.previous_timestamp = None

    def update(self, frame, timestamp, detections=None):
        """detections=None: không chạy AI; []: AI đã chạy nhưng không thấy đối tượng."""
        if self.previous_timestamp is not None and timestamp <= self.previous_timestamp:
            raise ValueError("Timestamp tracking phải tăng theo thời gian video.")
        dt = timestamp - self.previous_timestamp if self.previous_timestamp is not None else 0
        self.previous_timestamp = timestamp
        boxes = {t["id"]: bbox(t["x"]) for t in self.tracks
                 if timestamp - t["last_seen"] <= self.config["max_prediction_seconds"]}
        motion = self.motion.update(frame, boxes, foreground_flow=detections is None) if self.config["camera_motion"] else {
            "scene_cut": False, "reliable": False, "local_boxes": {}}
        self.last_motion = motion
        if motion["scene_cut"]:
            self.tracks.clear()
        self.tracks = [t for t in self.tracks if timestamp - t["last_seen"] <= self.config["max_lost_seconds"]]
        for t in self.tracks:
            t["source"] = "prediction"
            if motion["reliable"]:
                matrix = motion["matrix"]
                t["x"][:4] = measurement(warp_box(bbox(t["x"]), matrix))
                scale = np.hypot(matrix[0, 0], matrix[1, 0])
                transform = np.eye(8)
                transform[:2, :2] = transform[4:6, 4:6] = matrix[:, :2]
                transform[2, 2] = transform[3, 3] = transform[6, 6] = transform[7, 7] = scale
                t["x"][4:] = transform[4:, 4:] @ t["x"][4:]
                t["p"] = transform @ t["p"] @ transform.T
                t["source"] = "camera_motion"
            anchor = t["x"][:4].copy()
            transition = np.eye(8)
            transition[:4, 4:] = np.eye(4) * dt
            t["x"] = transition @ t["x"]
            t["p"] = transition @ t["p"] @ transition.T + np.eye(8) * max(dt, .001) * 4
            if t["id"] in motion["local_boxes"]:
                observed = measurement(motion["local_boxes"][t["id"]])
                if dt > 0:
                    t["x"][4:6] = .5 * t["x"][4:6] + .5 * (observed[:2] - anchor[:2]) / dt
                t["x"][:4] = observed
                t["source"] = "optical_flow"
        if detections is not None:
            costs = association_costs(self.tracks, detections, self.config)
            matched_tracks, matched_detections = set(), set()
            if costs.size:
                rows, cols = linear_sum_assignment(costs)
                for i, j in zip(rows, cols):
                    if costs[i, j] >= 1e5:
                        continue
                    t, row = self.tracks[i], detections[j]
                    gain = np.linalg.solve(t["p"][:4, :4] + np.eye(4) * 4, t["p"][:4, :]).T
                    t["x"] += gain @ (measurement(row["bbox_xyxy"]) - t["x"][:4])
                    residual = np.eye(8)
                    residual[:, :4] -= gain
                    t["p"] = residual @ t["p"] @ residual.T + 4 * gain @ gain.T
                    t["observed_bbox"] = list(row["bbox_xyxy"])
                    t.update(last_seen=timestamp, confidence=row["confidence"], source="detection", hits=t["hits"] + 1)
                    t["confirmed"] = t["confirmed"] or t["hits"] >= self.config["min_hits"]
                    matched_tracks.add(i)
                    matched_detections.add(j)
            for i, t in enumerate(self.tracks):
                if i not in matched_tracks and not t["confirmed"]:
                    t["hits"] = 0
            for j, row in enumerate(detections):
                if j in matched_detections:
                    continue
                self.tracks.append({"id": self.next_id, "label": row["label"],
                                    "x": np.r_[measurement(row["bbox_xyxy"]), np.zeros(4)], "p": np.eye(8) * 10,
                                    "last_seen": timestamp, "hits": 1, "confirmed": self.config["min_hits"] == 1,
                                    "confidence": row["confidence"], "observed_bbox": list(row["bbox_xyxy"]), "source": "detection"})
                self.next_id += 1
        output = []
        height, width = frame.shape[:2]
        for t in self.tracks:
            age = timestamp - t["last_seen"]
            if not t["confirmed"] or age > self.config["max_prediction_seconds"]:
                continue
            box = np.clip(bbox(t["x"]), [0, 0, 0, 0], [width, height, width, height])
            if np.any(box[2:] <= box[:2]) or not np.isfinite(box).all():
                continue
            output.append({"track_id": t["id"], "label": t["label"], "confidence": t["confidence"],
                           "bbox_xyxy": box.tolist(), "position_source": t["source"], "detection_age_seconds": age,
                           "observed_bbox_xyxy": t["observed_bbox"]})
        return output
