# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "ultralytics==8.3.200", "tensorflow==2.19.0", "tf-keras==2.19.0",
#   "onnx==1.17.0", "onnx2tf==1.28.2", "onnxslim>=0.1.67",
#   "onnxruntime", "onnx-graphsurgeon", "sng4onnx", "ai-edge-litert==1.3.0",
# ]
# ///
"""Export a rectangular YOLO11s model with float16 weights and float32 I/O."""
import argparse
from functools import partial
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11s.pt"))
    parser.add_argument("--output", type=Path, default=Path("android/app/src/main/assets/models/yolo11s.tflite"))
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(512, 1408))
    args = parser.parse_args()
    if any(n <= 0 or n % 32 for n in args.size):
        parser.error("Model height and width must be positive multiples of 32")
    weights, output = args.weights.resolve(), args.output.resolve()
    if not weights.is_file():
        parser.error(f"Missing weights: {weights}")
    with TemporaryDirectory(prefix="farol-export-") as directory:
        stage = Path(directory)
        config = stage / "ultralytics"
        config.mkdir()
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        # This Ultralytics release uses the legacy ONNX exporter. New PyTorch defaults to dynamo.
        torch.onnx.export = partial(torch.onnx.export, dynamo=False)
        local_weights = stage / "yolo11s.pt"
        shutil.copyfile(weights, local_weights)
        working_directory = Path.cwd()
        try:
            os.chdir(stage)
            exported = Path(YOLO(str(local_weights)).export(
                format="tflite", imgsz=list(args.size), half=True, nms=False, device="cpu", opset=17,
            )).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(exported, output)
        finally:
            os.chdir(working_directory)
    print(output)


if __name__ == "__main__":
    main()
