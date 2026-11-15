"""Converts an alert log into .srt subtitles to watch along with the video.

    uv run python -m farol.subtitles output/drive_alerts.json --lang pt-BR
    uv run python -m farol.subtitles            # converts every *_alerts.json in output/

Texts are rendered again from the message keys, so the same log can be subtitled in any language.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .i18n import DEFAULT, Translator, available

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"danger": "#ff4040", "warning": "#ffd040", "info": "#60e060"}


def _timestamp(s: float) -> str:
    ms = max(0, round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def srt(alerts: list[dict], tr: Translator, duration: float = 3.0, colors: bool = True) -> str:
    """Each alert stays on screen for `duration` seconds or until the next one starts."""
    alerts = sorted(alerts, key=lambda a: a["t"])
    blocks = []
    for i, a in enumerate(alerts):
        end = a["t"] + duration
        if i + 1 < len(alerts):
            end = min(end, alerts[i + 1]["t"] - 0.05)
        text = f"[{tr('level.' + a['level'])}] {tr.alert(a['key'], a.get('object'))}"
        if colors:
            text = f'<font color="{COLORS[a["level"]]}">{text}</font>'
        blocks.append(f"{i + 1}\n{_timestamp(a['t'])} --> {_timestamp(end)}\n{text}\n")
    return "\n".join(blocks)


def target_for(log: Path) -> Path:
    """`x_alerts.json` -> `x_farol.srt`: same name as the video, so players load it automatically."""
    return log.with_name(f"{log.stem.removesuffix('_alerts')}_farol.srt")


def convert(log: Path, tr: Translator, duration: float = 3.0, colors: bool = True) -> Path:
    target = target_for(log)
    target.write_text(srt(json.loads(log.read_text(encoding="utf-8")), tr, duration, colors),
                      encoding="utf-8")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", type=Path, nargs="*", help="*_alerts.json files (default: all in output/)")
    p.add_argument("--lang", default=DEFAULT, choices=available(), help="subtitle language")
    p.add_argument("--duration", type=float, default=3.0, help="seconds each alert stays on screen")
    p.add_argument("--no-colors", action="store_true", help="plain text, no colour per level")
    a = p.parse_args()
    logs = a.logs or sorted((ROOT / "output").glob("*_alerts.json"))
    if not logs:
        raise SystemExit("no *_alerts.json found")
    tr = Translator(a.lang)
    for log in logs:
        print(convert(log, tr, a.duration, not a.no_colors))


if __name__ == "__main__":
    main()
