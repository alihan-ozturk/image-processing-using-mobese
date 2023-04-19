import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.torch_utils import select_device, time_synchronized, TracedModel

from sort import *

import subprocess as sp
from utils.click import click


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]

        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    cudnn.benchmark = True  # set True to speed up constant image size inference
    # dataset = LoadCCTV(img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################
    cameraSize = (352, 640, 3)
    size = cameraSize[0] * cameraSize[1] * cameraSize[2]
    pipe = sp.Popen(["C:/ffmpeg/bin/ffmpeg.exe", "-i", "https://hls.ibb.gov.tr/tkm4/hls/31.stream/chunklist.m3u8",
                     "-loglevel", "quiet",
                     "-an",
                     "-f", "image2pipe",
                     "-pix_fmt", "bgr24",
                     "-vcodec", "rawvideo", "-"],
                    stdin=sp.PIPE, stdout=sp.PIPE)

    raw_image = pipe.stdout.read(size)
    img0 = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)
    event = click(img0, "alihan.txt", saveConfig=True)
    path = "cctv"

    # warm up
    # for i in range(3):
    #     model(img0s, augment=opt.augment)[0]
    pipe = sp.Popen(["C:/ffmpeg/bin/ffmpeg.exe", "-i", "https://hls.ibb.gov.tr/tkm4/hls/31.stream/chunklist.m3u8",
                     "-loglevel", "quiet",
                     "-an",
                     "-f", "image2pipe",
                     "-pix_fmt", "bgr24",
                     "-vcodec", "rawvideo", "-"],
                    stdin=sp.PIPE, stdout=sp.PIPE)

    while True:
        # preprocess
        raw_image = pipe.stdout.read(size)
        img0s = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)

        time.sleep(1)
        img1p = cv2.bitwise_and(img0s, img0s, mask=event.mask)
        img = cv2.add(img1p, event.vis)
        cv2.addWeighted(img1p, 0.5, img0s, 0.5, 0, img0s)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        # pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

    #     for i, det in enumerate(pred):  # detections per image
    #         p, s, img0, frame = path, '', img0s, getattr(dataset, 'frame', 0)
    #
    #         p = Path(p)  # to Path
    #         save_path = str(save_dir / p.name)  # img.jpg
    #         txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class
    #                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #
    #             dets_to_sort = np.empty((0, 6))
    #             # NOTE: We send in detected object class too
    #             for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
    #                 dets_to_sort = np.vstack((dets_to_sort,
    #                                           np.array([x1, y1, x2, y2, conf, detclass])))
    #
    #             if opt.track:
    #
    #                 tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
    #                 tracks = sort_tracker.getTrackers()
    #
    #                 # draw boxes for visualization
    #                 if len(tracked_dets) > 0:
    #                     bbox_xyxy = tracked_dets[:, :4]
    #                     identities = tracked_dets[:, 8]
    #                     categories = tracked_dets[:, 4]
    #                     confidences = None
    #
    #                     if opt.show_track:
    #                         # loop over tracks
    #                         for t, track in enumerate(tracks):
    #                             track_color = colors[int(track.detclass)] if not opt.unique_track_color else \
    #                                 sort_tracker.color_list[t]
    #
    #                             [cv2.line(im0, (int(track.centroidarr[i][0]),
    #                                             int(track.centroidarr[i][1])),
    #                                       (int(track.centroidarr[i + 1][0]),
    #                                        int(track.centroidarr[i + 1][1])),
    #                                       track_color, thickness=opt.thickness)
    #                              for i, _ in enumerate(track.centroidarr)
    #                              if i < len(track.centroidarr) - 1]
    #             else:
    #                 bbox_xyxy = dets_to_sort[:, :4]
    #                 identities = None
    #                 categories = dets_to_sort[:, 5]
    #                 confidences = dets_to_sort[:, 4]
    #
    #             im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
    #
    #         # Print time (inference + NMS)
    #         print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    #
    #         # Stream results
    #         ######################################################
    #         if dataset.mode != 'image' and opt.show_fps:
    #             currentTime = time.time()
    #
    #             fps = 1 / (currentTime - startTime)
    #             startTime = currentTime
    #             cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    #
    #         #######################################################
    #         if True:
    #             cv2.imshow(str(p), im0)
    #             cv2.waitKey(1)  # 1 millisecond
    #
    #         # Save results (image with detections)
    #         if save_img:
    #             vid_path = save_path
    #             if isinstance(vid_writer, cv2.VideoWriter):
    #                 vid_writer.release()  # release previous video writer
    #             fps, w, h = 30, im0.shape[1], im0.shape[0]
    #             save_path += '.mp4'
    #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #     vid_writer.write(im0)
    #
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     # print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='cctv', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--track', type=bool, default=True, help='run tracking')
    parser.add_argument('--show-track', type=bool, default=True, help='show tracked path')
    parser.add_argument('--show-fps', type=bool, default=False, help='show fps')
    parser.add_argument('--thickness', type=int, default=1, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=7,
                        min_hits=4,
                        iou_threshold=0.3)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
