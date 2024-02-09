import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
FILE = Path(__file__).resolve()

print(FILE)
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = os.path.join(ROOT, 'yolov5')
print(ROOT, type(ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (
    cv2,
    non_max_suppression,
    scale_boxes,
)
from utils.augmentations import (letterbox)

from utils.torch_utils import select_device



def load_model(weights, device=""):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    return model

def perform_inference(model, img_path):
    # Read the input image
    img = cv2.imread(img_path)
    o_img = img.copy()
    if img is None:
        print(f"Error: Unable to read image from {img_path}")
        return None
      
    
    # Inference
    img_size=640
    stride=32
    im = letterbox(img, img_size, stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(im).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

    
    pred = model(img, augment=False)

    # NMS  classes=[2,5,6,7]
    pred = non_max_suppression(pred, conf_thres=0.25,classes=[2,5,6,7], iou_thres=0.45)

    # Process predictions
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], o_img.shape).round()

            widths = det[:, 2] - det[:, 0]
            indices = torch.argmax(widths)

            x0, y0, x1, y1, conf, cls = det[indices]

            # Save the cropped image
            crop_img = o_img[int(y0):int(y1), int(x0):int(x1)]
            return crop_img

        else:
            return None

# Example usage in another file:


# model = load_model("yolov5s.pt", device="cpu")
# a = perform_inference(model, "../JEM207_Project/raw_data/BMW 116_2007/0.png")
# print(a)