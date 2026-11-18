"""Generate Android parity crops using the desktop OpenCV perception functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from farol.perception import (
    TRAFFIC_LIGHT,
    lamp_by_position,
    lit_pixel_counts,
    seen_from_behind,
    thirds_brightness,
    traffic_light_color,
)

TIMES = {
    "ida": (151.9, 191.9, 200.5, 203.1, 209.6, 226.6, 228.6, 231.4, 233.3, 236.2),
    "volta": (146.8, 152.2, 154.3),
}


def input_frames(folder):
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Could not read {path}")
            yield path.stem, frame
        elif path.suffix.lower() == ".mp4":
            trip = path.stem.rsplit(" ", 1)[-1]
            if trip not in TIMES:
                continue
            video = cv2.VideoCapture(str(path))
            try:
                for time in TIMES[trip]:
                    video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
                    ok, frame = video.read()
                    if not ok:
                        raise ValueError(f"Could not read {path} at {time}s")
                    yield f"{trip}-{time:.1f}", frame
            finally:
                video.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = YOLO(args.model, task="detect")
    records = []
    for label, frame in input_frames(args.frames):
        result = model.predict(frame, imgsz=args.imgsz, conf=.30, classes=[TRAFFIC_LIGHT], verbose=False)[0]
        for index, box in enumerate(result.boxes.xyxy.int().tolist()):
            x1, y1, x2, y2 = box
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if not crop.size:
                continue
            filename = f"{label}-{index}.png"
            if not cv2.imwrite(str(args.output / filename), crop):
                raise OSError(f"Could not write {filename}")
            records.append(dict(
                file=filename,
                lit_pixel_counts=lit_pixel_counts(frame, box),
                thirds_brightness=thirds_brightness(frame, box),
                lamp_by_position=lamp_by_position(frame, box),
                seen_from_behind=bool(seen_from_behind(frame, box)),
                traffic_light_color=traffic_light_color(frame, box),
            ))
    if not records:
        parser.error("No traffic-light boxes found")
    (args.output / "fixtures.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} traffic-light crops")


if __name__ == "__main__":
    main()
