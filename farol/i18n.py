"""Alert texts per language. Code deals in message keys; `locales/<lang>.json` holds the words.

To add a language, copy `locales/en.json` to `locales/<lang>.json` and translate the values.
Keys missing from a language fall back to English.
"""
from __future__ import annotations

import json
from pathlib import Path

LOCALES = Path(__file__).resolve().parent / "locales"
DEFAULT = "en"


def available() -> list[str]:
    return sorted(p.stem for p in LOCALES.glob("*.json"))


def _load(lang: str) -> dict[str, str]:
    path = LOCALES / f"{lang}.json"
    if not path.exists():
        raise SystemExit(f"unknown language '{lang}' (available: {', '.join(available())})")
    return json.loads(path.read_text(encoding="utf-8"))


class Translator:
    def __init__(self, lang: str = DEFAULT):
        self.lang = lang
        self.messages = _load(DEFAULT) if lang == DEFAULT else {**_load(DEFAULT), **_load(lang)}

    def __call__(self, key: str, **params) -> str:
        return self.messages[key].format(**params)

    def alert(self, key: str, params: dict | None = None) -> str:
        """Text of an alert. `params` holds language-neutral values that are translated here:
        `object` is an object kind ("pedestrian"), `lights` a list of (position, colour) pairs."""
        p = dict(params or {})
        if "object" in p:
            p["object"] = self(f"object.{p['object']}")
        if "lights" in p:
            p["lights"] = ", ".join(
                self("light.at", position=self(f"position.{pos}"), color=self(f"color.{color}"))
                for pos, color in p["lights"])
        return self(key, **p)
