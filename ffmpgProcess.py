import subprocess as sp
import time
import cv2
import numpy as np
import datetime

FFMPEG_BIN = "C:/ffmpeg/bin/ffmpeg.exe"
WEBURL = "https://hls.ibb.gov.tr/"
path = "imgs/"

cameraNames = ["BESIKTAS SEHITLER TEPESI",
               "BAR. BUL. YILDIZ",
               "ALIBEYKOY SILAHTARAGA TUNEL"
               "HARBIYE ALT GECIDI NISANTASI GIRIS",
               "DAVUTPASA",
               "S. YOLU CEVIZLI",
               "KIZ KULESI",
               "SPOR AKADEMISI",
               "SULTANBEYLI CARSI"]

cameraPaths = ["tkm4/hls/539.stream/chunklist.m3u8",
               "tkm4/hls/31.stream/chunklist.m3u8",
               "tkm2/hls/914.stream/chunklist.m3u8",
               "tkm2/hls/1022.stream/chunklist.m3u8",
               "tkm1/hls/171.stream/chunklist.m3u8",
               "tkm1/hls/261.stream/chunklist.m3u8",
               "tkm1/hls/214.stream/chunklist.m3u8",
               "tkm2/hls/447.stream/chunklist.m3u8",
               "tkm4/hls/284.stream/chunklist.m3u8"]

cameraSizes = [(352, 640, 3),
               (352, 640, 3),
               (352, 640, 3),
               (288, 352, 3),
               (1080, 1920, 3),
               (1080, 1920, 3),
               (720, 1280, 3),
               (1080, 1920, 3)]

cv2.namedWindow("test")
c = True
while c:
    for cameraName, cameraPath, cameraSize in zip(cameraNames, cameraPaths, cameraSizes):
        VIDEO_URL = WEBURL + cameraPath
        pipe = sp.Popen([FFMPEG_BIN, "-i", VIDEO_URL,
                         "-loglevel", "quiet",  # no text output
                         "-an",  # disable audio
                         "-f", "image2pipe",
                         "-pix_fmt", "bgr24",
                         "-vcodec", "rawvideo", "-"],
                        stdin=sp.PIPE, stdout=sp.PIPE)
        size = cameraSize[0] * cameraSize[1] * cameraSize[2]
        for i in range(101):
            if i % 100 == 0:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                raw_image = pipe.stdout.read(size)
                try:
                    image = np.frombuffer(raw_image, dtype='uint8').reshape(cameraSize)
                except ValueError:
                    print(cameraName, now)
                # imgName = path + cameraName + now + ".png"  # x
                # cv2.imwrite(imgName, image)  # x
                cv2.imshow("test", image)
                key = cv2.waitKey(70)
                if key == 27:
                    cv2.destroyAllWindows()
                    c = False
                    break
                elif key == ord("q"):
                    break
        # time.sleep(100)  # x
        if c == 0:
            break
