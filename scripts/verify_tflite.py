"""Validate the Android model's raw tensor contract against YOLO11s PyTorch."""
import argparse

import cv2
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from ultralytics import YOLO


def channels_first(output):
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected a single batch, got {output.shape}")
    if output.shape[1] == 84:
        return output[0]
    if output.shape[2] == 84:
        return output[0].T
    raise ValueError(f"Expected raw COCO scores and boxes, got {output.shape}")


def intersection_over_union(a, b):
    a1, b1 = a[:2] - a[2:] / 2, b[:2] - b[2:] / 2
    a2, b2 = a[:2] + a[2:] / 2, b[:2] + b[2:] / 2
    intersection = np.prod(np.maximum(0, np.minimum(a2, b2) - np.maximum(a1, b1)))
    return intersection / (np.prod(a[2:]) + np.prod(b[2:]) - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--band", default=.65, type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runtime = Interpreter(model_path=args.model, num_threads=4)
    runtime.allocate_tensors()
    input_info, = runtime.get_input_details()
    output_info, = runtime.get_output_details()
    assert input_info["dtype"] == output_info["dtype"] == np.float32, "Expected float32 I/O"
    batch, height, width, channels = input_info["shape"]
    assert (batch, channels) == (1, 3), "Expected NHWC RGB"
    frame = cv2.imread(args.image)
    if frame is None:
        parser.error(f"Could not read {args.image}")
    assert 0 < args.band <= 1
    crop = frame[:max(1, int(frame.shape[0] * args.band))]
    scale = min(width / crop.shape[1], height / crop.shape[0])
    w, h = round(crop.shape[1] * scale), round(crop.shape[0] * scale)
    left, top = (width - w) // 2, (height - h) // 2
    image = np.full((1, height, width, 3), 114, dtype=np.uint8)
    image[0, top:top + h, left:left + w] = cv2.resize(crop, (w, h))[..., ::-1]
    image = image.astype(np.float32) / 255
    runtime.set_tensor(input_info["index"], image)
    runtime.invoke()
    raw = runtime.get_tensor(output_info["index"])
    android = channels_first(raw).copy()
    android[:4] *= np.array([width, height, width, height])[:, None]
    model = YOLO(args.weights).model.eval()
    with torch.inference_mode():
        desktop = channels_first(model(torch.from_numpy(image).permute(0, 3, 1, 2))[0].numpy())
    assert android.shape == desktop.shape, (android.shape, desktop.shape)
    print(f"Input {tuple(input_info['shape'])}; output {raw.shape}; normalized XYWH")
    verified = 0
    for cls in (9, 11):
        scores = desktop[4 + cls]
        index = int(np.argmax(scores))
        if scores[index] < .30:
            continue
        other = int(np.argmax(android[4 + cls]))
        overlap = intersection_over_union(desktop[:4, index], android[:4, other])
        print(f"COCO {cls}: PT {scores[index]:.3f}, TFLite {android[4 + cls, other]:.3f}, box IoU {overlap:.3f}")
        assert overlap >= .70 and abs(scores[index] - android[4 + cls, other]) <= .15
        verified += 1
    assert verified, "Use an image with a visible traffic light or stop sign"


if __name__ == "__main__":
    main()
