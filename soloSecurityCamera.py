import subprocess as sp
import cv2 as cv
import numpy as np
import datetime
from sklearn.cluster import KMeans
from utils import click

FFMPEG_BIN = "C:/ffmpeg/bin/ffmpeg.exe"
WEBURL = "https://hls.ibb.gov.tr/"
path = "./"

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

cameraName = "BESIKTAS SEHITLER TEPESI"
cameraPath = "tkm4/hls/539.stream/chunklist.m3u8"
cameraSize = (352, 640, 3)

cv.namedWindow("test")
VIDEO_URL = WEBURL + cameraPath
pipe = sp.Popen([FFMPEG_BIN, "-i", VIDEO_URL,
                 "-loglevel", "quiet",
                 "-an",
                 "-f", "image2pipe",
                 "-pix_fmt", "bgr24",
                 "-vcodec", "rawvideo", "-"],
                stdin=sp.PIPE, stdout=sp.PIPE)

trackLen = 2
tracks = []

size = cameraSize[0] * cameraSize[1] * cameraSize[2]
raw_image = pipe.stdout.read(size)
img = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)
event = click(img, "alihan.txt", saveConfig=True)

masked = cv.bitwise_and(img, img, mask=event.mask)
maskC0 = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
maskC0 = cv.GaussianBlur(maskC0, (5, 5), 1)

while True:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    raw_image = pipe.stdout.read(size)
    try:
        img = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)
    except ValueError:
        print(cameraName, now)
        continue

    vis = img.copy()
    masked = cv.bitwise_and(img, img, mask=event.mask)
    maskC1 = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
    maskC1 = cv.GaussianBlur(maskC1, (5, 5), 1)
    cv.addWeighted(masked, 0.5, vis, 1 - 0.5, 0, vis)
    if len(tracks) > 0:
        # print(len(tracks))
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv.calcOpticalFlowPyrLK(maskC0, maskC1, p0, None, **lk_params)
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(maskC1, maskC0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []

        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > trackLen:
                del tr[0]
            new_tracks.append(tr)
            # print(x, y)
            cv.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        tracks = new_tracks

    for x, y in [np.int32(tr[-1]) for tr in tracks]:
        cv.circle(vis, (int(x), int(y)), 3, 0, -1)

    p = cv.goodFeaturesToTrack(maskC1, mask=event.mask, **feature_params)
    if p is not None:
        for x, y in np.float32(p).reshape(-1, 2):
            tracks.append([(x, y)])

    maskC0 = maskC1
    cv.imshow("test", vis)
    key = cv.waitKey(1)
    if key == 27:
        cv.destroyAllWindows()
        break
    elif key == ord("s"):
        imgName = path + cameraName + now + ".png"
        cv.imwrite(imgName, img)
