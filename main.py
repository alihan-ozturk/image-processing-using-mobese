import json
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
from sort import *


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]

        # if not opt.nobbox:
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        # if not opt.nolabel:
        label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(4)]
names = ["bus", "shuttle", "motorcycle", "car"]

model = attempt_load("./best.pt", map_location="cuda")
model = TracedModel(model, "cuda", 640)

m3u8_url = "https://hls.ibb.gov.tr/tkm1/hls/492.stream/chunklist.m3u8"
streamSize = (1080, 1920, 3)

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

ret, thresh = cv2.threshold(maskC.mask, 0, 255, cv2.THRESH_BINARY)
n = len(maskC.masks)

sort_tracker = Sort(max_age=5,
                    min_hits=2,
                    iou_threshold=0.2)

while True:
    im0 = thread.lastFrame.copy()

    img = cv2.bitwise_and(im0, im0, mask=thresh)
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

    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = ""
        # if webcam:  # batch_size >= 1
        #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        # else:
        #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        #
        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            dets_to_sort = np.empty((0, 6))
            # NOTE: We send in detected object class too
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort,
                                          np.array([x1, y1, x2, y2, conf, detclass])))

            # if opt.track:

            tracked_dets = sort_tracker.update(dets_to_sort, True)
            tracks = sort_tracker.getTrackers()

            # draw boxes for visualization
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                confidences = None

                # if opt.show_track:
                #     # loop over tracks
                for t, track in enumerate(tracks):
                    track_color = colors[int(track.detclass)]  # if not True else sort_tracker.color_list[t]

                    [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])),
                              (int(track.centroidarr[i + 1][0]),
                               int(track.centroidarr[i + 1][1])),
                              track_color, thickness=1)
                     for i, _ in enumerate(track.centroidarr)
                     if i < len(track.centroidarr) - 1]
            # else:
            #     bbox_xyxy = dets_to_sort[:, :4]
            #     identities = None
            #     categories = dets_to_sort[:, 5]
            #     confidences = dets_to_sort[:, 4]

            im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

        # Print time (inference + NMS)
        # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Stream results
        ######################################################
        # if dataset.mode != 'image' and opt.show_fps:
        #     currentTime = time.time()
        #
        #     fps = 1 / (currentTime - startTime)
        #     startTime = currentTime
        #     cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        #######################################################
        # if view_img:
        # cv2.imshow("frame", im0)
        # cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'image':
        #         cv2.imwrite(save_path, im0)
        #         print(f" The image with the result is saved in: {save_path}")
        #     else:  # 'video' or 'stream'
        #         if vid_path != save_path:  # new video
        #             vid_path = save_path
        #             if isinstance(vid_writer, cv2.VideoWriter):
        #                 vid_writer.release()  # release previous video writer
        #             if vid_cap:  # video
        #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             else:  # stream
        #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                 save_path += '.mp4'
        #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #         vid_writer.write(im0)

    # for i, det in enumerate(pred):
    #     if len(det):
    #         det[:, :4] = scale_coords(inputSize[:2], det[:, :4], streamSize[:2]).round()
    # nv = {i: 0 for i in range(n+1)}
    # for *xyxy, conf, cls in det:
    #     x1, y1, x2, y2 = xyxy
    #     xx, yy = int((x2 - (x2 - x1) / 2).cpu().numpy()), int((y2 - (y2 - y1) / 2).cpu().numpy())
    #
    #     nv[maskC.mask[yy, xx]] += 1
    #     label = f'{names[int(cls)]} {conf:.2f}'
    #     plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
    #
    # image = cv2.putText(img0, json.dumps(nv), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #                     1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("frame", im0)
    if cv2.waitKey(1) == key:
        cv2.destroyAllWindows()
        break

thread.stop()
