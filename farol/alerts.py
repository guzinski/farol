"""Decides what to say and when: priority, minimum repetition interval and speech synthesis."""
from __future__ import annotations

import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DANGER, WARNING, INFO = 0, 1, 2
LEVEL_NAMES = {DANGER: "danger", WARNING: "warning", INFO: "info"}


@dataclass
class Alert:
    t: float
    level: int
    key: str  # message key, e.g. "traffic_light.red"
    params: dict | None = None  # language-neutral values for the message, see Translator.alert

    def record(self, tr) -> dict:
        return {"t": self.t, "level": LEVEL_NAMES[self.level], "key": self.key,
                "params": self.params, "text": tr.alert(self.key, self.params)}


class AlertManager:
    """Avoids chatter: each kind of warning has its own interval and there is a minimum
    silence between utterances. Danger jumps the queue."""

    def __init__(self, min_silence: float = 1.8):
        self.min_silence = min_silence
        self.last_by_slot: dict[str, float] = {}
        self.last_t = -1e9
        self.emitted: list[Alert] = []

    def try_emit(self, t: float, slot: str, level: int, key: str, params: dict | None = None,
                 *, repeat_after: float) -> bool:
        if t - self.last_by_slot.get(slot, -1e9) < repeat_after:
            return False
        gap = 0.9 if level == DANGER else self.min_silence
        if t - self.last_t < gap:
            return False
        self.last_by_slot[slot] = t
        self.last_t = t
        self.emitted.append(Alert(round(t, 2), level, key, params))
        return True


class Voice:
    """Offline synthesis with Piper. Each sentence is generated once and reused."""

    def __init__(self, model: Path, cache: Path):
        self.model, self.cache = model, cache
        cache.mkdir(parents=True, exist_ok=True)

    def wav(self, text: str) -> Path:
        name = "".join(c if c.isalnum() else "_" for c in text.lower())[:80] + ".wav"
        target = self.cache / name
        if not target.exists():
            subprocess.run(
                [sys.executable, "-m", "piper", "-m", str(self.model), "-f", str(target)],
                input=text.encode(), check=True, capture_output=True,
            )
        return target

    def track(self, utterances: list[tuple[float, str]], duration_s: float, target: Path) -> Path:
        """Builds a WAV with each (time, text) utterance at the moment the device would say it."""
        rate = None
        pieces = []
        for t, text in utterances:
            with wave.open(str(self.wav(text))) as w:
                rate = rate or w.getframerate()
                pieces.append((t, np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)))
        rate = rate or 22050
        end = max([duration_s] + [t + len(p) / rate for t, p in pieces])
        audio = np.zeros(int(end * rate) + 1, dtype=np.int32)
        for t, p in pieces:
            i = int(t * rate)
            audio[i:i + len(p)] += p
        with wave.open(str(target), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(np.clip(audio, -32768, 32767).astype(np.int16).tobytes())
        return target
