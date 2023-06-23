import time
import threading
import subprocess as sp
import numpy as np


class CustomThread(threading.Thread):
    def __init__(self, m3u8_url, cameraSize, fps):
        threading.Thread.__init__(self)
        self.pipe = sp.Popen(["C:/ffmpeg/bin/ffmpeg.exe", "-i", m3u8_url,
                              "-loglevel", "quiet",
                              "-an",
                              "-f", "image2pipe",
                              "-pix_fmt", "bgr24",
                              "-vcodec", "rawvideo", "-"],
                             stdin=sp.PIPE, stdout=sp.PIPE)
        self.size = cameraSize[0] * cameraSize[1] * cameraSize[2]
        self.cameraSize = cameraSize
        self.fps = fps
        self.c = True
        self.lastFrame = None

    def run(self):
        while self.c:

            raw_image = self.pipe.stdout.read(self.size)
            time.sleep(self.fps)
            try:
                self.lastFrame = np.frombuffer(raw_image, dtype='uint8').reshape(self.cameraSize)
            except ValueError:
                pass
        return

    def stop(self):
        self.c = False
        print("stopped")