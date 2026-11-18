# farol Android app — v1 specification

Hand-off document for implementing the first version of the farol phone app. Read this whole
file, then `farol/perception.py`, `farol/__main__.py` (`evaluate`), `farol/alerts.py`,
`farol/i18n.py` and `farol/locales/*.json`: the app is a port of that pipeline to live camera
input. Do not change the desktop pipeline's behaviour while doing this.

## 1. Goal

A **local debug build** (sideloaded APK, not published) that the author can mount on the
dashboard and test on real streets in Porto Alegre, Brazil:

- live camera → detect traffic lights and stop signs → read the light colour → speak alerts in
  Brazilian Portuguese, with the same rules the desktop version uses;
- record each drive (video + alert log + per-frame detections) so it can be reviewed later on
  the desktop and compared with `python -m farol` on the same footage;
- a replay mode that runs the same pipeline on a video file, so changes can be tested without
  driving.

Priority is detection quality, not battery, heat or app size. Those are optimised later.

**Target device: Samsung Galaxy S25+ only** (Snapdragon 8 Elite for Galaxy: Adreno 830 GPU,
Hexagon NPU, 12 GB RAM; 50 MP main camera, 3× telephoto; Android 15+ / One UI 7+). v1 does not
need to run on any other phone, so pick device-specific options whenever they help, and skip
compatibility fallbacks unless they are needed on this phone.

## 2. Non-goals for v1

- Publishing to Google Play; iOS.
- Vehicle / pedestrian warnings (the desktop `--obstacles` mode is off and stays out of v1).
- Brazilian regulatory sign classification and traffic light arrow recognition (next versions;
  see section 12).
- Optical zoom / switching to the telephoto lens.

## 3. Platform decisions

| Topic | Decision | Why |
| --- | --- | --- |
| OS | Android only, minSdk 35 (the S25+ ships with Android 15), targetSdk latest stable | Single target phone; iOS needs a Mac |
| Language / UI | Kotlin, Jetpack Compose | Standard |
| Camera | CameraX (`Preview` + `ImageAnalysis` + `VideoCapture`), Camera2 interop for exposure | Simplest reliable API with the controls we need |
| Model runtime | LiteRT (TensorFlow Lite). **Step 1:** GPU delegate (Adreno 830), float16. **Step 2:** Qualcomm QNN delegate on the Hexagon NPU | GPU float16 matches the desktop model most closely; the NPU is much faster but needs quantisation |
| Model | YOLO11s COCO, exported by Ultralytics to `.tflite` | Same model as desktop, so results are comparable |
| Image processing | Plain Kotlin on small crops (no OpenCV dependency) | The colour logic only touches boxes of a few hundred pixels |
| Speech | Android `TextToSpeech`, locale `pt-BR` | Built in; no Piper needed |
| Location in repo | `android/` at the repo root, its own Gradle project | Keeps the Python project untouched |

Keep the model runner behind an interface (`ModelRunner`) so GPU and NPU runners can be
switched from the debug panel and compared on the same drive.

**NPU (step 2, after everything else works on the GPU):** the Hexagon NPU runs quantised
models. Use the Qualcomm QNN delegate for LiteRT (`com.qualcomm.qti:qnn-litert-delegate`, or
whatever Qualcomm currently publishes for LiteRT; check Qualcomm AI Hub, which also offers
YOLO models already compiled for Snapdragon 8 Elite). Quantise with calibration images from the
author's own trip frames, not COCO only: dashcam frames with small lights. Prefer w8a16 (8-bit
weights, 16-bit activations) if the delegate supports it, since full int8 tends to lose small
objects. Accept the NPU model only if, on the parity frames of §12, it finds the same traffic
lights as the GPU model (box IoU ≥ 0.7). Small lights are exactly what quantisation tends to
lose. What the NPU buys is a bigger input (§5) at the same frame rate.

## 4. Camera and the "zoom" (high-resolution crops)

Most desktop errors come from lights being only ~15×40 px in a 1080p frame. The phone sensor
has far more detail, so v1 uses two resolutions:

1. `ImageAnalysis` asks for the **largest resolution the device supports up to 3840×2160**
   (`ResolutionSelector` with `ResolutionStrategy(Size(3840,2160), FALLBACK_RULE_CLOSEST_LOWER)`),
   `STRATEGY_KEEP_ONLY_LATEST`, output format YUV_420_888.
2. The **detector** gets a downscaled copy of the **top part of the frame only** (see §5).
3. Every detected box is mapped back to full-resolution coordinates and the **colour is read from
   the full-resolution crop** (§6). Convert only the crop from YUV to RGB, never the whole 4K
   frame.

The S25+ main camera delivers 4K YUV at 30 fps, so 4K analysis is the default. A 1080p
analysis setting exists only for comparison in the debug panel. Use the main (wide) camera, not
the logical multi-camera, so the lens never switches by itself.

Exposure: washed-out red lamps are partly the camera exposing for the whole scene. Add:

- an **exposure compensation** setting (`CameraControl.setExposureCompensationIndex`), default
  −1 EV, adjustable in the debug panel;
- an optional toggle "meter on lights": a `FocusMeteringAction` with an AE metering point on the
  upper-centre region of the image (where lights are), refreshed every few seconds.

Lock the frame rate to 30 fps (`Camera2Interop` `CONTROL_AE_TARGET_FPS_RANGE` = [30,30]) so LED
lights flicker less between frames. Keep autofocus continuous; focus at infinity is acceptable
if continuous AF hunts.

`VideoCapture` records the drive at 1080p in parallel (see §9). If the S25+ refuses to bind all
three use cases at these resolutions, drop video recording to 720p before reducing analysis
resolution. Check with `CameraInfo`/`isUseCaseCombinationSupported` at startup and log the
combination actually bound.

## 5. Detection

### Model export

```sh
uv run --with tensorflow --with onnx --with onnxslim python -c \
  "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='tflite', imgsz=[512,1408], half=True)"
```

`imgsz=[height,width]` must be multiples of 32. The detection band of a 16:9 frame (top 65 %)
has an aspect of about 2.7:1. Start with `[512,1408]` on the GPU. On the desktop, more input
pixels was the single biggest improvement for small lights (960 → 1600 square), so once the
NPU runner works, try `[704,1920]` and keep the largest size that sustains ≥ 15 analysed
frames/s. Make the model file **loadable from the app's files directory** (`adb push` a new
`.tflite`) so sizes can be tried without rebuilding. Read the input
size from the model's input tensor, never hard-code it.

Verify the output layout by running the exported model in Python on one test image and
comparing with the `.pt` model before writing the Kotlin decoder. Ultralytics TFLite output is
normally `[1, 84, N]` (cx, cy, w, h, then 80 class scores), with coordinates normalised to the
input size. Confirm this rather than assume it.

### Region analysed

Lights and signs that matter are above the hood. Feed the detector the band from the top of the
frame down to `0.65 × frame height` (the desktop already ignores lights whose bottom edge is
below `0.65·h`), resized with letterboxing to the model input. Keep `0.65` as a setting.

### Post-processing

- Keep only COCO classes 9 (traffic light) and 11 (stop sign); confidence ≥ 0.30.
- Class-wise NMS, IoU 0.45 (implement in Kotlin; do not rely on the export including NMS).
- Map boxes back to full-frame, full-resolution pixel coordinates.

### Tracking

The desktop uses ByteTrack. For v1 implement a simple greedy IoU tracker: match each new box to
the previous frame's box of the same class with the highest IoU ≥ 0.3, else assign a new id;
drop tracks unseen for 10 analysed frames. Ids are needed for the stop sign rule (§7).

### Frame rate

Analyse up to 15 frames/s (desktop default). If the phone is slower, process whatever arrives
(latest-only) and use real timestamps. The smoothing windows in §6–7 count *analysed frames*,
so log the effective analysis rate in every session.

## 6. Traffic light colour — exact port of `farol/perception.py`

Port these functions **behaviour for behaviour**. HSV must match OpenCV's 8-bit convention:
H in 0–179 (degrees / 2), S and V in 0–255, V = max(R,G,B), S = 255·(V−min)/V.

All thresholds below are from the current desktop code and must be named constants.

**`lit_pixel_counts(box)`** — returns per-colour counts and the box area, or null if the crop
height < 8 px.
- lit pixel: S > 90 and V > 160
- red: H < 10 or H > 168; yellow: 14 ≤ H ≤ 34; green: 42 ≤ H ≤ 98
- if crop height ≥ 2 × width (vertical light), split rows into thirds (`linspace(0,h,4)`
  boundaries) and count red only in the top third, yellow only in the middle, green only in the
  bottom.

**`thirds_brightness(box)`** — mean V of top / middle / bottom thirds (`array_split` semantics:
the first parts get the extra rows); null if height < 6.

**`lamp_by_position(box)`** — only if height ≥ 2 × width: the brightest third wins
(top = red, middle = yellow, bottom = green) if it beats the second brightest by ≥ 25; else null.

**`seen_from_behind(box)`** — Porto Alegre signal housings are painted yellow; from behind the
box is mostly that yellow. True when the fraction of pixels with 12 ≤ H ≤ 35, S > 40, V > 90 is
> 0.25 **and** the fraction with V < 90 is < 0.25.

**`traffic_light_color(box)`**:
1. null if `lit_pixel_counts` is null or `seen_from_behind`;
2. take the colour with the most lit pixels; if that count < max(4, 0.02 × area), return
   `lamp_by_position` (may be null);
3. else return that colour.

Because v1 reads colour from crops that are 2–4× larger than the desktop's, the absolute pixel
thresholds behave slightly differently. Keep them identical in v1 and record the numbers per
detection (§9) so they can be retuned from real data.

### Per-position tracking — `TrafficLightsByPosition`

Port as is (default mode on desktop):
- positions: left if box centre x < 0.38·W, right if > 0.62·W, else ahead
- candidates: lights with a colour and bottom edge < 0.65·H; drop lights shorter than
  0.5 × the tallest candidate (they belong to the next junction)
- per position keep the tallest candidate
- per position a vote window of 9 analysed frames (null votes count); a colour becomes that
  position's state when it has ≥ 6 of 9 votes and the window is full
- overall state: null if no position has a colour; a single colour if all agree; otherwise the
  ordered list of (position, colour) pairs.

Known weakness, keep it in v1 but log it: image position ≠ the movement the light controls
(a left-turn light hanging over the left lanes can sit in the middle of the image).

## 7. Alert rules — port of `evaluate()` and `AlertManager`

Traffic light (state compared with the previous frame's state):
- state became a list of pairs → `traffic_light.by_position` with `lights` = the pairs, level
  WARNING, slot `light-<state>`, repeat_after 3 s
- state became a colour → key `traffic_light.turned_green` if it went red → green, else
  `traffic_light.<colour>`; level WARNING for red/yellow, INFO for green; repeat_after 3 s
- state unchanged and red → `traffic_light.red` again, slot `light-red`, repeat_after 30 s

Stop sign: a detection with confidence > 0.45 and red fraction > 0.35 (red: H < 10 or H > 168,
S > 80, V > 60, over the whole box) increments a counter for its track id; on the 4th frame emit
`sign.stop`, WARNING, slot `stop`, repeat_after 8 s.

`AlertManager.try_emit`: refuse if the slot fired less than `repeat_after` ago, or if any alert
fired less than 1.8 s ago (0.9 s for DANGER); otherwise record it.

## 8. Text and speech

- Copy `farol/locales/en.json` and `pt-BR.json` into the app's assets **unchanged** and port
  `Translator`: lookup with English fallback, `{param}` substitution, `lights` rendered as
  `light.at` strings joined by ", " with `position.*` and `color.*` translated. Default language
  pt-BR, selectable.
- `TextToSpeech` with `Locale("pt","BR")`; speak each emitted alert with `QUEUE_FLUSH` (a newer
  light state replaces an old one that is still being spoken). Request transient audio focus
  with ducking so music lowers instead of stopping. Speech rate setting in the debug panel.
- If no pt-BR voice is installed, show a message telling the user to install one in Android's
  text-to-speech settings.

## 9. Session recording

Each drive is a session in `Android/media/<app id>/sessions/<yyyy-MM-dd_HH-mm-ss>/` (visible
to `adb pull` and file managers):

- `video.mp4` — 1080p recording of the same camera;
- `alerts.json` — **same schema as the desktop** `*_alerts.json`: a list of
  `{t, level, key, params, text}` with `t` in seconds from the start of the video, so
  `python -m farol.subtitles alerts.json` produces subtitles for `video.mp4`;
- `detections.jsonl` — one line per analysed frame: `t`, and for each detection the class,
  conf, track id, full-res box, colour, position, lit pixel counts, thirds brightness,
  `seen_from_behind`;
- `session.json` — device model, analysis resolution, model file and input size, effective
  analysis fps, exposure settings, app version (git commit).

Timestamps must share one clock with the video. Use the camera frame timestamps relative to the
recording start, not wall-clock time.

## 10. Screens

1. **Disclaimer** on first launch (and from settings): experiment, not a safety device, do not
   rely on it to drive; the person driving must not operate the phone while moving.
2. **Drive screen** (landscape, full screen, keep screen on):
   - camera preview with boxes: each light coloured by its read colour, labelled with colour
     and position; stop signs in red;
   - three large circles top-left for left / ahead / right state (as the desktop video draws);
   - the last alert as large high-contrast text, 3 s;
   - one big start/stop session button; recording indicator.
3. **Debug panel** (slide-over, also usable before driving): analysis fps, inference ms,
   analysis resolution, model file picker (from the files directory), exposure compensation,
   "meter on lights" toggle, detection band height (0.65), speech on/off and rate, language.
4. **Replay**: pick a video file; run the full pipeline on its frames (no camera), show the same
   overlay, and write the same session files. Use it to compare with the desktop on the trip
   videos.

The UI is for a sighted helper during testing; the end product for low vision users is the
speech. Keep the drive screen simple.

## 11. Repository layout and build

```
android/
  settings.gradle.kts, build.gradle.kts, gradle/ (version catalog), gradlew
  app/src/main/java/io/github/guzinski/farol/
    camera/      CameraX setup, YUV crop/convert, exposure
    detect/      ModelRunner (interface), LiteRtRunner, letterbox, decode, NMS, IouTracker
    perception/  LightColor.kt (§6 functions), TrafficLightsByPosition.kt, StopSign.kt
    rules/       Evaluator.kt (§7), AlertManager.kt
    i18n/        Translator.kt
    speech/      Speaker.kt
    session/     SessionRecorder.kt (§9)
    replay/      VideoFrameSource.kt
    ui/          DriveScreen, DebugPanel, ReplayScreen, Disclaimer
  app/src/main/assets/locales/{en,pt-BR}.json, models/yolo11s.tflite
  app/src/test/  JVM unit tests for perception, rules, i18n, NMS, tracker
```

`.tflite` files follow the existing `*.onnx`/`*.pt` rule and are not committed. Add `*.tflite`
to `.gitignore` and document the export command in `android/README.md`.

Build and install: build from **Windows** (Android Studio or `gradlew.bat installDebug`), not
WSL, because USB `adb` from WSL needs extra setup. On the phone, enable developer options and
USB debugging. `android/README.md` must list the exact steps, including pushing a model:
`adb push yolo11s.tflite /sdcard/Android/media/<app id>/models/`.

## 12. Tests

- **Parity fixtures (most important)**: write a small Python script (e.g. `scripts/fixtures.py`)
  that runs the desktop functions on the frames in a folder and writes, for each traffic light
  box, the crop as PNG plus the expected `lit_pixel_counts`, `thirds_brightness`,
  `lamp_by_position`, `seen_from_behind` and `traffic_light_color`. Kotlin unit tests load the
  same PNGs and must match exactly (thirds brightness within 0.5). Useful frames: the trip
  videos at 151.9, 191.9, 200.5, 203.1, 209.6, 226.6, 228.6, 231.4, 233.3 and 236.2 s (ida) and
  146.8, 152.2 and 154.3 s (volta). These include washed-out red, LED green read by lamp
  position, yellow housings seen from behind, and dark/unreadable lights.
- Unit tests for `TrafficLightsByPosition` vote sequences, `AlertManager` timing, `Translator`
  output for `traffic_light.by_position`, NMS, and the IoU tracker.
- Replay of `teste viagem boulervard ida.mp4` and `volta.mp4` (author's videos, not in the repo)
  should give an alert list close to `output/1600b/` on the desktop. Differences are expected
  from model size and runtime; record them, don't hide them.

## 13. Acceptance criteria for v1

- Installs on the author's Galaxy S25+ from a local build following `android/README.md`.
- Drive screen runs ≥ 15 analysed frames/s on 4K analysis frames with the default model (GPU),
  and logs the actual rate.
- Speaks pt-BR alerts for light changes and stop signs, with the desktop rules.
- Each session produces the four files of §9, and `python -m farol.subtitles` turns
  `alerts.json` into subtitles that line up with `video.mp4`.
- Replay mode works on the two trip videos.
- Parity tests pass.

## 14. Known problems carried over, and what comes after v1

Carried over from the desktop (see the per-frame analysis in the project history):
- position (left/ahead/right) comes from the image, not from the movement the light controls;
- some green LED lights are unreadable when the three thirds have similar brightness;
- distant lights of the next junction can still be counted.

Next versions, in order:
1. **Arrow plates and arrow lamps**: Porto Alegre junctions put a small plate under each light
   (↑ straight, ↰ left / U-turn). A classifier on the high-res crop under each light gives the
   real movement and replaces image position.
2. **Regulatory signs (R-1 … R-40)**: two-stage, a sign detector plus a classifier on the
   high-res crop; same crop machinery as the arrows.
3. Optical zoom with the telephoto lens for lights that stay unreadable; battery, heat and
   model size optimisation; then publishing.

## 15. Open questions for the author

- Phone mount position: centred behind the rear-view mirror is assumed; the band and position
  thresholds assume the horizon near the middle of the image.
- The S25+ will get hot on a sunny dashboard while running camera + model + recording. If it
  throttles, the session log shows the analysis fps dropping; a vent mount helps.
