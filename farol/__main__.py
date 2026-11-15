"""Runs farol over a recorded video and produces an annotated video + subtitles + alert log.

usage: uv run python -m farol videos/drive.mp4 [--lang pt-BR] [--start 0 --duration 60]
                                               [--obstacles] [--audio]

By default only traffic lights and signs are announced; --obstacles turns on vehicle and
pedestrian warnings (still too many false alarms) and --audio also renders the spoken track.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import unicodedata
from pathlib import Path

import cv2

from .alerts import DANGER, INFO, WARNING, AlertManager, Voice
from .i18n import DEFAULT, Translator, available
from .perception import (BICYCLE, MOTORCYCLE, PERSON, STOP_SIGN, TRAFFIC_LIGHT, VEHICLES, VULNERABLE,
                         Corridor, Detector, RelevantTrafficLight, TTCEstimator, red_fraction, riding,
                         traffic_light_color)
from .subtitles import convert

ROOT = Path(__file__).resolve().parent.parent
BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0), None: (160, 160, 160)}
LEVEL_BGR = {DANGER: (0, 0, 255), WARNING: (0, 200, 255), INFO: (255, 255, 255)}
LIGHT_LEVEL = {"red": WARNING, "yellow": WARNING, "green": INFO}


def evaluate(t, frame, dets, corridor, ttc, light, alerts, stop_seen, obstacles=False):
    """Rules: turns what was perceived into warnings."""
    # --- Traffic light: speaks when the state changes
    before = light.state
    now = light.update([d for d in dets if d.cls == TRAFFIC_LIGHT])
    if now and now != before:
        key = "traffic_light.turned_green" if (before, now) == ("red", "green") else f"traffic_light.{now}"
        alerts.try_emit(t, f"light-{now}", LIGHT_LEVEL[now], key, repeat_after=3)
    elif now == "red":
        alerts.try_emit(t, "light-red", WARNING, "traffic_light.red", repeat_after=30)

    # --- Stop sign: must show up in a few frames
    for d in dets:
        if d.cls == STOP_SIGN and d.conf > 0.45 and red_fraction(frame, d.box) > 0.35:
            stop_seen[d.id] = stop_seen.get(d.id, 0) + 1
            if stop_seen[d.id] == 4:
                alerts.try_emit(t, "stop", WARNING, "sign.stop", repeat_after=8)

    # --- Obstacles in the way (off by default: it fires on parked cars, cars in other
    #     lanes and cross traffic)
    if not obstacles:
        return
    worst = None  # (urgency, detection, kind)
    mounts = [m.box for m in dets if m.cls in (MOTORCYCLE, BICYCLE)]
    for d in dets:
        if d.cls in (TRAFFIC_LIGHT, STOP_SIGN):
            continue
        if d.cls == PERSON and any(riding(d.box, m) for m in mounts):
            continue  # the rider of the motorcycle/bicycle, not a pedestrian
        d.in_corridor = corridor.contains(d.box)
        d.ttc = ttc.update(t, d)
        if not d.in_corridor:
            continue
        prox = corridor.proximity(d.box)
        kind = VULNERABLE.get(d.cls) or VEHICLES.get(d.cls)
        vulnerable = d.cls in (PERSON, BICYCLE)  # motorcycles weave through traffic: strong alert only if closing in
        if (d.ttc is not None and d.ttc < 2.0 and prox > 0.35) or (vulnerable and prox > 0.6):
            urgency = DANGER
        elif (d.ttc is not None and d.ttc < 4.0 and prox > 0.25) or (vulnerable and prox > 0.2) \
                or (d.cls == MOTORCYCLE and prox > 0.5):
            urgency = WARNING
        else:
            continue
        if worst is None or (urgency, -prox) < (worst[0], -corridor.proximity(worst[1].box)):
            worst = (urgency, d, kind)
    if worst:
        urgency, d, kind = worst
        if urgency == DANGER:
            alerts.try_emit(t, f"danger-{kind}", DANGER, "obstacle.close", kind, repeat_after=4)
        else:
            alerts.try_emit(t, f"warning-{kind}", WARNING, "obstacle.ahead", kind, repeat_after=6)
    ttc.prune({d.id for d in dets})


def ascii_only(text: str) -> str:
    """OpenCV's built-in font has no accented characters."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def draw(frame, dets, corridor, light_state, recent, t, tr):
    cv2.polylines(frame, [corridor.polygon], True, (255, 200, 0), 2)
    for d in dets:
        x1, y1, x2, y2 = d.box
        if d.cls == TRAFFIC_LIGHT:
            color, label = BGR[d.color], tr(f"color.{d.color}") if d.color else "?"
        elif d.cls == STOP_SIGN:
            color, label = (0, 0, 255), tr("label.stop")
        else:
            color = (0, 140, 255) if d.in_corridor else (120, 120, 120)
            kind = VULNERABLE.get(d.cls) or VEHICLES.get(d.cls)
            label = (tr(f"object.{kind}") if kind else "") + (f" {d.ttc:.1f}s" if d.ttc and d.in_corridor else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, ascii_only(label), (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.circle(frame, (40, 40), 24, BGR[light_state], -1)
    cv2.circle(frame, (40, 40), 24, (255, 255, 255), 2)
    y = 40
    for a in recent:
        if t - a.t < 3.0:
            text = ascii_only(tr.alert(a.key, a.obj))
            cv2.putText(frame, text, (80, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(frame, text, (80, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, LEVEL_BGR[a.level], 2)
            y += 38


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--lang", default=DEFAULT, choices=available(), help="language of the alerts")
    p.add_argument("--start", type=float, default=0, help="first second to process")
    p.add_argument("--duration", type=float, default=None, help="seconds to process")
    p.add_argument("--model", default=str(ROOT / "yolo11s.pt"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--horizon", type=float, default=0.55, help="height of the horizon in the image (0-1)")
    p.add_argument("--hood", type=float, default=0.92, help="where the car's hood starts (0-1)")
    p.add_argument("--fps", type=float, default=15, help="frames per second analysed")
    p.add_argument("--out", type=Path, default=ROOT / "output", help="output folder")
    p.add_argument("--obstacles", action="store_true", help="turn on vehicle and pedestrian warnings")
    p.add_argument("--audio", action="store_true", help="render the spoken track and mux it into the video")
    a = p.parse_args()

    tr = Translator(a.lang)
    voice = None
    if a.audio:
        voice_model = ROOT / "voices" / f"{tr('meta.voice')}.onnx"
        if not voice_model.exists():
            raise SystemExit(f"Piper voice not found: {voice_model} (download it or run without --audio)")
        voice = Voice(voice_model, ROOT / "voices" / "cache" / a.lang)

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {a.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(video_fps / a.fps))
    fps = video_fps / step
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)

    a.out.mkdir(parents=True, exist_ok=True)
    name = a.video.stem
    raw = a.out / f"{name}_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = Detector(a.model, a.imgsz)
    corridor = Corridor(w, h, a.horizon, a.hood)
    ttc, light, alerts = TTCEstimator(), RelevantTrafficLight(w, h), AlertManager()
    stop_seen: dict[int, int] = {}

    n, t0 = 0, time.time()
    while True:
        for _ in range(step - 1):
            cap.grab()
        ok, frame = cap.read()
        t = n / fps
        if not ok or (a.duration and t >= a.duration):
            break
        dets = detector(frame)
        for d in dets:
            if d.cls == TRAFFIC_LIGHT:
                d.color = traffic_light_color(frame, d.box)
        evaluate(t, frame, dets, corridor, ttc, light, alerts, stop_seen, a.obstacles)
        draw(frame, dets, corridor, light.state, alerts.emitted[-3:], t, tr)
        writer.write(frame)
        n += 1
        if n % 150 == 0:
            print(f"  {t:6.1f}s  {n / (time.time() - t0):.1f} frames/s  {len(alerts.emitted)} alerts", flush=True)
    cap.release()
    writer.release()
    speed = n / (time.time() - t0)

    records = [x.record(tr) for x in alerts.emitted]
    log = a.out / f"{name}_alerts.json"
    log.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    subs = convert(log, tr)
    final = a.out / f"{name}_farol.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw)]
    if voice:
        track = voice.track([(r["t"], r["text"]) for r in records], n / fps, a.out / f"{name}_speech.wav")
        cmd += ["-i", str(track), "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(final)]
    subprocess.run(cmd, check=True)
    raw.unlink()
    print(f"\n{n} frames at {speed:.1f} frames/s (analysing {fps:.0f}/s) - {len(records)} alerts")
    for r in records:
        print(f"  {r['t']:6.1f}s  {tr('level.' + r['level']):8s} {r['text']}")
    print(f"\nvideo:     {final}" + ("  (with speech)" if voice else ""))
    print(f"subtitles: {subs}")


if __name__ == "__main__":
    main()
