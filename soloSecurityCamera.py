import subprocess as sp
import cv2
import numpy as np
import datetime
from utils import click

FFMPEG_BIN = "C:/ffmpeg/bin/ffmpeg.exe"
WEBURL = "https://hls.ibb.gov.tr/"
path = "./"

cameraName = "BESIKTAS SEHITLER TEPESI"
cameraPath = "tkm4/hls/539.stream/chunklist.m3u8"
cameraSize = (352, 640, 3)

cv2.namedWindow("test")
VIDEO_URL = WEBURL + cameraPath
pipe = sp.Popen([FFMPEG_BIN, "-i", VIDEO_URL,
                 "-loglevel", "quiet",
                 "-an",
                 "-f", "image2pipe",
                 "-pix_fmt", "bgr24",
                 "-vcodec", "rawvideo", "-"],
                stdin=sp.PIPE, stdout=sp.PIPE)
size = cameraSize[0] * cameraSize[1] * cameraSize[2]
raw_image = pipe.stdout.read(size)
img = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)
event = click(img, "alihan.txt", saveConfig=True)
while True:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    raw_image = pipe.stdout.read(size)
    try:
        image = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)
    except ValueError:
        print(cameraName, now)
        continue
    imgC, imgD = event.applyMask(image)
    cv2.imshow("test", imgD)
    key = cv2.waitKey(70)
    if key == 27:
        cv2.destroyAllWindows()
        break
    elif key == ord("s"):
        imgName = path + cameraName + now + ".png"
        cv2.imwrite(imgName, image)
