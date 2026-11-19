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
    position: str | None = None  # traffic lights only: "left", "ahead" or "right" when it counts


def prefer_directml() -> bool:
    """Makes ONNX models run on the GPU through DirectML (any DirectX 12 GPU on Windows: AMD,
    Intel or NVIDIA). Ultralytics only picks CUDA, CoreML or Windows ML (Windows 11 24H2+) by
    itself, so on Windows 10 it would fall back to the CPU. Needs the `onnxruntime-directml`
    package; does nothing without it."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return False
    if getattr(ort.InferenceSession, "farol_directml", False):
        return True
    base = ort.InferenceSession

    class DirectMLSession(base):
        farol_directml = True

        def __init__(self, path, sess_options=None, providers=None, **kwargs):
            if providers == ["CPUExecutionProvider"]:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                sess_options = sess_options or ort.SessionOptions()
                sess_options.enable_mem_pattern = False  # not supported by DirectML
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            super().__init__(path, sess_options, providers=providers, **kwargs)

    ort.InferenceSession = DirectMLSession
    return True


class Detector:
    def __init__(self, model: str = "yolo11s.pt", imgsz: int = 1600, conf: float = 0.3):
        if str(model).endswith(".onnx"):
            prefer_directml()
        self.model = YOLO(model, task="detect")
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


def lit_pixel_counts(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[dict[str, int], int] | None:
    """Bright, saturated pixels of each lamp hue inside a traffic light box, and the box area.
    In a vertical light each colour only counts in its own lamp's third (red top, yellow middle,
    green bottom), so a yellow-painted housing or a reflection does not read as a lit lamp.
    None when the box is too small to read."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0 or crop.shape[0] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 160)
    hues = {
        "red": (h < 10) | (h > 168),
        "yellow": (h >= 14) & (h <= 34),
        "green": (h >= 42) & (h <= 98),
    }
    if crop.shape[0] >= 2 * crop.shape[1]:
        third = np.zeros(h.shape, dtype=np.uint8)
        bounds = np.linspace(0, h.shape[0], 4).astype(int)
        for i in range(3):
            third[bounds[i]:bounds[i + 1]] = i
        for i, color in enumerate(hues):
            hues[color] = hues[color] & (third == i)
    counts = {color: int((lit & mask).sum()) for color, mask in hues.items()}
    return counts, lit.size


def thirds_brightness(frame: np.ndarray, box) -> tuple[float, float, float] | None:
    """Mean brightness of the top, middle and bottom thirds of a box."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0 or crop.shape[0] < 6:
        return None
    v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    top, mid, bottom = np.array_split(v, 3, axis=0)
    return float(top.mean()), float(mid.mean()), float(bottom.mean())


def lamp_by_position(frame: np.ndarray, box, min_gap: float = 25) -> str | None:
    """Reads a vertical light by which lamp is lit: top = red, middle = yellow, bottom = green.
    Works when the colour itself is washed out (old incandescent lamps in direct sun, LED green
    that the camera renders almost white), as long as one third is clearly brighter."""
    x1, y1, x2, y2 = box
    if y2 - y1 < 2 * (x2 - x1):  # not a vertical light, or seen from the side
        return None
    thirds = thirds_brightness(frame, box)
    if thirds is None:
        return None
    ranked = sorted(range(3), key=lambda i: thirds[i], reverse=True)
    if thirds[ranked[0]] - thirds[ranked[1]] < min_gap:
        return None
    return ("red", "yellow", "green")[ranked[0]]


def seen_from_behind(frame: np.ndarray, box, max_housing: float = 0.25, min_dark: float = 0.25) -> bool:
    """True for a light seen from the back or the side. Porto Alegre paints its signal housings
    yellow: from behind the box is mostly that yellow, while from the front it is the dark face
    and lamps. Sunlight on a yellow back would otherwise read as a lit lamp."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    housing = ((h >= 12) & (h <= 35) & (s > 40) & (v > 90)).mean()
    return housing > max_housing and (v < 90).mean() < min_dark


def traffic_light_color(frame: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    """Classifies the lit lamp by the hue of the bright, saturated pixels, falling back to
    which lamp is lit when the hue is too washed out to tell."""
    read = lit_pixel_counts(frame, box)
    if read is None:
        return None
    counts, area = read
    minimum = max(4, 0.02 * area)
    if seen_from_behind(frame, box):
        # In direct sun the yellow frame around the face of a front-facing light also passes
        # the housing test. Its yellow could still be the housing, but enough lit red or green
        # that beats the yellow or matches the brighter lamp can only come from a lit lamp.
        color = max(("red", "green"), key=lambda c: counts[c])
        count = counts[color]
        if count >= minimum and (count > counts["yellow"] or lamp_by_position(frame, box) == color):
            return color
        return None
    color, count = max(counts.items(), key=lambda kv: kv[1])
    if count < minimum:
        return lamp_by_position(frame, box)
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

    @property
    def display(self) -> list[str | None]:
        return [self.state]

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


POSITIONS = ("left", "ahead", "right")


class TrafficLightsByPosition:
    """Tracks one colour per side of the image (left / ahead / right) instead of a single light,
    for junctions where each movement has its own signal.

    `state` is a colour name when every visible light agrees, a tuple of (position, colour)
    pairs when they differ, and None when no light is visible.
    """

    def __init__(self, width: int, height: int, frames: int = 9,
                 left_until: float = 0.38, right_from: float = 0.62, min_relative_height: float = 0.5):
        self.w, self.h = width, height
        self.left_until, self.right_from = left_until * width, right_from * width
        self.min_relative_height = min_relative_height
        self.votes = {p: deque(maxlen=frames) for p in POSITIONS}
        self.colors: dict[str, str | None] = {p: None for p in POSITIONS}
        self.state: str | tuple | None = None

    @property
    def display(self) -> list[str | None]:
        return [self.colors[p] for p in POSITIONS]

    def _position(self, box) -> str:
        x = (box[0] + box[2]) / 2
        return "left" if x < self.left_until else "right" if x > self.right_from else "ahead"

    def update(self, lights: list[Detection]) -> str | tuple | None:
        candidates = [s for s in lights if s.color and s.box[3] < 0.65 * self.h]
        if candidates:
            # much smaller lights belong to the next junction, not this one
            tallest = max(s.box[3] - s.box[1] for s in candidates)
            candidates = [s for s in candidates if s.box[3] - s.box[1] >= self.min_relative_height * tallest]
        best: dict[str, Detection] = {}
        for s in candidates:
            s.position = self._position(s.box)
            other = best.get(s.position)
            if other is None or s.box[3] - s.box[1] > other.box[3] - other.box[1]:
                best[s.position] = s
        for p in POSITIONS:
            votes = self.votes[p]
            votes.append(best[p].color if p in best else None)
            color, count = Counter(votes).most_common(1)[0]
            if count >= (votes.maxlen * 2) // 3 and len(votes) == votes.maxlen:
                self.colors[p] = color
        seen = [(p, self.colors[p]) for p in POSITIONS if self.colors[p]]
        if not seen:
            self.state = None
        elif len({c for _, c in seen}) == 1:
            self.state = seen[0][1]
        else:
            self.state = tuple(seen)
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
