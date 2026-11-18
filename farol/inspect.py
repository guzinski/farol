"""Shows what farol sees in a still image: every traffic light and stop sign, with the numbers
behind the colour decision. A debugging aid for single frames or photos.

    uv run python -m farol.inspect photo.jpg [more.jpg ...]

For each image it prints one line per detection and writes `<image>_inspect.jpg` next to it
(or in --out), with the boxes drawn and each traffic light enlarged in a strip at the bottom.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .perception import (STOP_SIGN, TRAFFIC_LIGHT, Detection, Detector, TrafficLightsByPosition, lamp_by_position,
                         lit_pixel_counts, red_fraction, seen_from_behind, thirds_brightness,
                         traffic_light_color)

ROOT = Path(__file__).resolve().parent.parent
BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0), None: (200, 200, 200)}


def zoom_strip(frame: np.ndarray, lights, height: int = 160) -> np.ndarray | None:
    tiles = []
    for i, d in enumerate(lights, 1):
        x1, y1, x2, y2 = d.box
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        scale = height / crop.shape[0]
        tile = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), height), interpolation=cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(tile, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(tile, f"#{i}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR[d.color], 2)
        tiles.append(tile)
    if not tiles:
        return None
    strip = np.hstack(tiles)
    if strip.shape[1] < frame.shape[1]:
        strip = cv2.copyMakeBorder(strip, 0, 0, 0, frame.shape[1] - strip.shape[1], cv2.BORDER_CONSTANT,
                                   value=(30, 30, 30))
    else:
        strip = cv2.resize(strip, (frame.shape[1], int(strip.shape[0] * frame.shape[1] / strip.shape[1])))
    return strip


def inspect(path: Path, detector: Detector, out_dir: Path | None) -> Path:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"could not read image {path}")
    h, w = frame.shape[:2]
    r = detector.model.predict(frame, classes=[TRAFFIC_LIGHT, STOP_SIGN], imgsz=detector.imgsz,
                               conf=detector.conf, verbose=False)[0]
    dets = [Detection(i, c, conf, tuple(b)) for i, (b, c, conf) in enumerate(zip(
        r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()))]
    lights = sorted((d for d in dets if d.cls == TRAFFIC_LIGHT), key=lambda d: d.box[0])
    positions = TrafficLightsByPosition(w, h)
    print(f"\n{path.name}  ({w}x{h})  {len(lights)} traffic light(s), "
          f"{sum(d.cls == STOP_SIGN for d in dets)} stop sign(s)")
    for i, d in enumerate(lights, 1):
        d.color = traffic_light_color(frame, d.box)
        d.position = positions._position(d.box)
        x1, y1, x2, y2 = d.box
        read = lit_pixel_counts(frame, d.box)
        counts = " ".join(f"{k}={v}" for k, v in read[0].items()) if read else "too small"
        thirds = thirds_brightness(frame, d.box)
        thirds_txt = "/".join(f"{t:.0f}" for t in thirds) if thirds else "-"
        by_lamp = lamp_by_position(frame, d.box) or "-"
        ignored = " (ignored: low in image)" if y2 >= 0.65 * h else ""
        if seen_from_behind(frame, d.box):
            ignored += " (ignored: seen from behind)"
        print(f"  #{i}  {d.color or '?':6s} {d.position:5s}  box {x2 - x1}x{y2 - y1} px at ({x1},{y1})  "
              f"conf {d.conf:.2f}  lit pixels: {counts}  brightness top/mid/bottom: {thirds_txt} "
              f"(lit lamp: {by_lamp}){ignored}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), BGR[d.color], 2)
        cv2.putText(frame, f"#{i} {d.color or '?'}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    BGR[d.color], 2)
    for d in dets:
        if d.cls == STOP_SIGN:
            x1, y1, x2, y2 = d.box
            print(f"  STOP  box {x2 - x1}x{y2 - y1} px  conf {d.conf:.2f}  red fraction {red_fraction(frame, d.box):.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    strip = zoom_strip(frame, lights)
    if strip is not None:
        frame = np.vstack([frame, strip])
    target = (out_dir or path.parent) / f"{path.stem}_inspect.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), frame)
    print(f"  -> {target}")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", default=str(ROOT / "yolo11s.pt"))
    p.add_argument("--imgsz", type=int, default=1600)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=None, help="folder for the annotated images (default: next to each)")
    a = p.parse_args()
    detector = Detector(a.model, a.imgsz, a.conf)
    for img in a.images:
        inspect(img, detector, a.out)


if __name__ == "__main__":
    main()
