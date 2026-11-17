# farol

Prototype of a camera-based driving assistant that calls out traffic lights and signs, aimed
at people with low vision. *Farol* is Portuguese for traffic light, headlight and lighthouse.

It runs over a recorded dashcam video and produces an annotated video, `.srt` subtitles with
each alert, and a JSON alert log. Alerts can also be rendered as speech.

> **This is an experiment, not a safety device.** It misses things and raises false alarms.
> Do not rely on it to drive.

## What it detects

- Traffic lights: red, yellow, green, and the change from red to green. At junctions with a
  separate signal per movement, lights on the left, ahead and right are tracked separately and
  announced together when they differ ("left red, ahead green")
- Stop signs
- Vehicles, pedestrians, cyclists and motorcycles in the path ahead — off by default
  (`--obstacles`), because it still fires on parked cars, other lanes and cross traffic

## Usage

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg`. The YOLO model is
downloaded on first run.

```sh
uv sync
uv run python -m farol videos/drive.mp4 --lang pt-BR
```

Results go to `output/`: `drive_farol.mp4`, `drive_farol.srt` and `drive_alerts.json`. The
subtitle file has the same name as the video, so most players load it automatically.

| Option | Meaning |
| --- | --- |
| `--lang` | Language of the alerts (`en`, `pt-BR`) |
| `--start`, `--duration` | Process only part of the video (seconds) |
| `--fps` | Frames per second analysed (default 15) |
| `--horizon`, `--hood` | Where the horizon and the car's hood sit in the image (0-1) |
| `--lights` | `position` (default) tracks left / ahead / right lights; `single` only the most relevant one |
| `--model` | Detection model; an `.onnx` file runs on the GPU through DirectML when available |
| `--obstacles` | Turn on vehicle and pedestrian warnings |
| `--audio` | Render the spoken track and mux it into the video |

`--audio` needs a [Piper](https://github.com/rhasspy/piper) voice in `voices/`, named in the
language file (`pt_BR-faber-medium.onnx` for `pt-BR`, `en_US-lessac-medium.onnx` for `en`),
together with its `.onnx.json`.

## GPU on Windows / WSL (AMD, Intel or NVIDIA)

On the CPU, farol processes about 3 frames/s of 1080p video. Any DirectX 12 GPU can run the
model through DirectML instead: on an RX 6750 XT that is about 15 frames/s, real time for the
15 frames/s farol analyses. WSL has no GPU access for most AMD cards, so this runs a Windows
Python from WSL. Setup is at the top of `scripts/farol-windows.sh`; then:

```sh
scripts/farol-windows.sh videos/drive.mp4 --lang pt-BR
```

Subtitles can be regenerated from a log in any language:

```sh
uv run python -m farol.subtitles output/drive_alerts.json --lang en
```

## Adding a language

Copy `farol/locales/en.json` to `farol/locales/<lang>.json` and translate the values. Keys
missing from a language fall back to English.
