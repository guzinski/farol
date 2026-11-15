"""Perception: detection/tracking (YOLO), traffic light colour and analysis of the corridor ahead."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes we care about
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK, TRAFFIC_LIGHT, STOP_SIGN = 0, 1, 2, 3, 5, 7, 9, 11
CLASSES = [PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK, TRAFFIC_LIGHT, STOP_SIGN]
# class -> object kind (translated through the "object.<kind>" message keys)
VULNERABLE = {PERSON: "pedestrian", BICYCLE: "cyclist", MOTORCYCLE: "motorcycle"}
VEHICLES = {CAR: "vehicle", BUS: "bus", TRUCK: "truck"}


@dataclass
class Detection:
    id: int
    cls: int
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    in_corridor: bool = False
    ttc: float | None = None  # estimated time to collision (s)
    color: str | None = None  # traffic lights only: "red", "yellow" or "green"


class Detector:
    def __init__(self, model: str = "yolo11s.pt", imgsz: int = 960, conf: float = 0.3):
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.track(
            frame, persist=True, classes=CLASSES, imgsz=self.imgsz, conf=self.conf,
            tracker="bytetrack.yaml", verbose=False,
        )[0]
        dets = []
        if r.boxes is None or r.boxes.id is None:
            return dets
        for box, cls, conf, tid in zip(
            r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(),
            r.boxes.conf.tolist(), r.boxes.id.int().tolist(),
        ):
            dets.append(Detection(tid, cls, conf, tuple(box)))
        return dets


def traffic_light_color(frame: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    """Classifies the lit lamp by the hue of the bright, saturated pixels."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0 or crop.shape[0] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 160)
    n = {
        "red": int((lit & ((h < 10) | (h > 168))).sum()),
        "yellow": int((lit & (h >= 14) & (h <= 34)).sum()),
        "green": int((lit & (h >= 42) & (h <= 98)).sum()),
    }
    color, count = max(n.items(), key=lambda kv: kv[1])
    if count < max(4, 0.02 * lit.size):
        return None
    return color


class Corridor:
    """Trapezoid ahead of the car: anything whose bottom edge is inside it is in the way."""

    def __init__(self, width: int, height: int, horizon: float = 0.55,
                 hood: float = 0.92, bottom_half_width: float = 0.30, top_half_width: float = 0.04,
                 center: float = 0.5):
        self.w, self.h = width, height
        c = center * width
        self.y_top, self.y_bottom = horizon * height, hood * height
        self.polygon = np.array([
            [c - bottom_half_width * width, self.y_bottom],
            [c - top_half_width * width, self.y_top],
            [c + top_half_width * width, self.y_top],
            [c + bottom_half_width * width, self.y_bottom],
        ], dtype=np.int32)

    def contains(self, box) -> bool:
        x1, _, x2, y2 = box
        y = min(y2, self.y_bottom - 1)
        # tests the centre and the 30%/70% points of the box's bottom edge
        return any(
            cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0
            for x in ((x1 + x2) / 2, x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7)
        )

    def proximity(self, box) -> float:
        """0 = at the horizon, 1 = touching the hood."""
        return float(np.clip((box[3] - self.y_top) / (self.y_bottom - self.y_top), 0, 1))


class TTCEstimator:
    """Time to collision from the box growth rate (no real distance needed)."""

    def __init__(self, window_s: float = 0.8):
        self.window = window_s
        self.history: dict[int, deque] = {}

    def update(self, t: float, det: Detection) -> float | None:
        x1, y1, x2, y2 = det.box
        size = float(np.sqrt(max(x2 - x1, 1) * max(y2 - y1, 1)))
        h = self.history.setdefault(det.id, deque())
        h.append((t, size))
        while h and t - h[0][0] > self.window:
            h.popleft()
        if len(h) < 4 or h[-1][0] - h[0][0] < self.window * 0.5:
            return None
        ts = np.array([p[0] for p in h])
        sizes = np.array([p[1] for p in h])
        rate = np.polyfit(ts, sizes, 1)[0]  # px/s
        if rate <= 0.03 * size:  # growing less than 3%/s: not approaching in a relevant way
            return None
        return float(size / rate)

    def prune(self, live_ids: set[int]):
        for k in [k for k in self.history if k not in live_ids]:
            del self.history[k]


class RelevantTrafficLight:
    """Picks the traffic light that applies to the driver and smooths its colour across frames."""

    def __init__(self, width: int, height: int, frames: int = 9):
        self.w, self.h = width, height
        self.votes: deque = deque(maxlen=frames)
        self.state: str | None = None

    def update(self, lights: list[Detection]) -> str | None:
        candidates = [s for s in lights if s.color and s.box[3] < 0.65 * self.h]
        chosen = None
        if candidates:
            def score(s: Detection) -> float:
                x1, y1, x2, y2 = s.box
                offset = abs((x1 + x2) / 2 - self.w / 2) / (self.w / 2)
                return (y2 - y1) * (1.0 - 0.6 * offset)  # bigger and more central
            chosen = max(candidates, key=score)
        self.votes.append(chosen.color if chosen else None)
        color, count = Counter(self.votes).most_common(1)[0]
        if count >= (self.votes.maxlen * 2) // 3 and len(self.votes) == self.votes.maxlen:
            self.state = color
        return self.state


def red_fraction(frame: np.ndarray, box) -> float:
    """A STOP sign is almost entirely red; a speed limit sign (red border, white middle) is not."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h < 10) | (h > 168)) & (s > 80) & (v > 60)).mean())


def riding(person, vehicle) -> bool:
    """The person is on the motorcycle/bicycle: centres aligned and boxes overlapping vertically."""
    overlap_x = min(person[2], vehicle[2]) - max(person[0], vehicle[0])
    narrower = max(1, min(person[2] - person[0], vehicle[2] - vehicle[0]))
    return overlap_x / narrower > 0.5 and person[3] > vehicle[1] and person[1] < vehicle[3]
