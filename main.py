import sys
import time
import cv2
import numpy as np

from utils.streamVideo import CustomThread, letterbox
from utils.click import click

import torch
from models.experimental import attempt_load
from utils.torch_utils import TracedModel
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import random

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(4)]
names = ["bus", "shuttle", "motorcycle", "car"]

model = attempt_load("./best.pt", map_location="cuda")
model = TracedModel(model, "cuda", 640)

m3u8_url = "https://hls.ibb.gov.tr/tkm4/hls/752.stream/chunklist.m3u8"
streamSize = (352, 640, 3)

key = ord("q")
thread = CustomThread(m3u8_url, streamSize, 0.07)
thread.start()

d = 4
print(f"wait {d} seconds")
time.sleep(d)
if thread.lastFrame is not None:
    maskC = click(thread.lastFrame, "mask.txt", saveConfig=True)
else:
    thread.stop()
    sys.exit()

while True:
    img0 = thread.lastFrame.copy()

    img = cv2.bitwise_and(img0, img0, mask=maskC.mask)
    img[maskC.mask == 0] = 114
    img = letterbox(img)
    inputSize = img.shape
    img = img[:, :, ::-1].transpose(2, 0, 1)

    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to("cuda").float()
    img /= 255.0
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(inputSize[:2], det[:, :4], streamSize[:2]).round()

            image = cv2.putText(img0, str(len(det)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            a = 0
            b = 0
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

    cv2.imshow("frame", img0)
    if cv2.waitKey(1) == key:
        cv2.destroyAllWindows()
        break

thread.stop()
