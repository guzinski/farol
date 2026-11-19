# Farol for Android

Farol v1 is a local debug app for the Samsung Galaxy S25+ on Android 15 or later. It analyses the rear wide camera, speaks traffic-light and stop-sign alerts, and records drives for comparison with desktop Farol. Use it as an experiment with a passenger operating the phone.

## Build and install from Windows

1. Open the `android/` directory of a Windows checkout in Android Studio. Install JDK 21, Android SDK Platform 37 and Android SDK Platform-Tools. If using PowerShell, put Java, Git, and `adb.exe` on `PATH`. Set `ANDROID_HOME` to the Windows SDK directory or let Android Studio create `android/local.properties`.
2. On the S25+, enable Developer options by tapping **Build number** seven times under **Settings → About phone → Software information**. Enable **USB debugging**, plug in the cable, and approve the computer's debugging fingerprint on the phone.
3. Run these commands from the repository root in Windows PowerShell.

   ```powershell
   adb devices -l
   cd android
   .\gradlew.bat :app:assembleDebug :app:installDebug
   ```

   The device should appear as `device`. The APK is at `android/app/build/outputs/apk/debug/app-debug.apk`. Android Studio's Run button also installs the debug build.
4. If the model is not bundled in the APK, open Farol once, then push the exported file from the repo root.

   ```powershell
   adb shell mkdir -p /sdcard/Android/media/io.github.guzinski.farol/models
   adb push android/app/src/main/assets/models/yolo11s.tflite /sdcard/Android/media/io.github.guzinski.farol/models/
   ```

   Open **Ajustes → Buscar modelos** and select `yolo11s.tflite`. A new file with a different name can be selected from the same menu. The app also reads models placed under its private `files/models/` directory.

Use Windows ADB for USB installation. The model export can run from Linux or WSL.

## Export YOLO11s

Place the desktop `yolo11s.pt` weights at the repository root and run the exporter from there.

```sh
uv run scripts/export_android_model.py
```

The script uses an isolated Ultralytics 8.3.200 environment and exports YOLO11s COCO as TFLite with `imgsz=[512,1408], half=True, nms=False`. This Ultralytics release supports the float16 TFLite export. The script writes `android/app/src/main/assets/models/yolo11s.tflite`. If the file is present at build time, the debug APK bundles it. You can use `--size HEIGHT WIDTH` for another shape or `--output PATH` to store the model elsewhere. Height and width must be multiples of 32. Generated weights are Git-ignored.

To validate a replacement model, pass a frame with a visible light or stop sign to the verification script.

```sh
uv run --with ai-edge-litert python scripts/verify_tflite.py \
  android/app/src/main/assets/models/yolo11s.tflite path/to/frame.png
```

This checks the float32 NHWC input and raw normalized XYWH output against PyTorch. The runner reads the input size at load time. **GPU · FP16** is the default. **CPU · debug** runs the same model on the CPU for comparison. The `ModelRunner` interface is shared by live and replay. Qualcomm QNN and calibration are the second phase after GPU measurements on the phone.

## Use and review

- Start a session to turn on live alerts and save the drive. **Ajustes** shows the effective analysis FPS, image preparation time, inference time, and frame size. Use **Testar voz** to check speech before starting a session. You can select the analysis resolution, exposure compensation, AE metering, detection band, language, and speech settings before a session.
- When analysed frames are more than 160 ms apart, two consecutive matching color readings at the same position can confirm a light. At that cadence the app keeps the confirmed color through one missing reading, which bumps and blur often cause, and clears it after a second missing reading in a row or a gap over 1.5 s.
- Select **Reproduzir vídeo** to pick a video from Android's file picker. Replay analyses up to 15 source frames per second. MP4 inputs, including the 10-bit HLG trip videos, are copied as-is for review. Other supported containers are saved as a video-only MP4. Live sessions record video without audio.
- Open **Histórico** to review saved drives and replays on the phone. Each session shows analysed frames, tracked detections, traffic-light colors, alerts, and the saved video. Tap an alert to jump to its timestamp. Existing sessions appear automatically.
- Sessions are stored under `/sdcard/Android/media/io.github.guzinski.farol/sessions/<yyyy-MM-dd_HH-mm-ss>/`. They contain `video.mp4`, `alerts.json`, `detections.jsonl`, and `session.json`. Live logs use the camera timestamps relative to the first encoded MP4 sample. Replay logs use decoded media presentation timestamps. `session.json` records both analysis throughput and the sampling rate on the video timeline.
- Pull sessions and generate subtitles from the repo root.

  ```powershell
  adb pull /sdcard/Android/media/io.github.guzinski.farol/sessions/ ./output/android-sessions/
  uv run python -m farol.subtitles ./output/android-sessions/<session>/alerts.json --lang pt-BR
  ```

  Open `video.mp4` and load `alerts_farol.srt` in your player. Compare replay alert keys and timestamps with the corresponding `output/1600b/*_alerts.json` files. Check `detections.jsonl` around differences. The detector input, capture resolution and tracker differ from desktop. Android's HDR decoder can also produce different RGB values from OpenCV. Light position in the log is the x-coordinate in the image.

## Tests

Run the checks from `android/` on Windows.

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:lintDebug
```

The parity test checks OpenCV lamp counts, thirds, brightness fallback, yellow housings and final color on 61 light crops from 13 frames of the two trip videos. To regenerate the fixtures, run the generator from the repo root with the original trip MP4s or source images in a folder.

```sh
PYTHONPATH=. uv run python scripts/fixtures.py /path/to/frames-or-trip-videos android/app/src/test/resources/parity
```

On the S25+, verify the bound stream combination with `adb logcat -s FarolCamera`, then make a short drive recording. Check `analysis_fps` in `session.json` against the 15 FPS target at 4K. `detections.jsonl` records preparation, inference and total frame times, plus the thermal status, thermal headroom and battery temperature every 2 s. `session.json` summarizes them. Compare subtitles with the recorded video, confirm Portuguese speech, and run both trip videos through replay.
