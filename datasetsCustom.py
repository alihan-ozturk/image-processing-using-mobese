import cv2
import numpy as np


class LoadCCTV:  # for inference
    def __init__(self, img_size=640, stride=32, cameraPath='tkm4/hls/31.stream/chunklist.m3u8',
                 FFMPEG_BIN='C:/ffmpeg/bin/ffmpeg.exe', cameraSize=(352, 640, 3)):
        import subprocess as sp
        from utils.click import click
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.cameraSize = cameraSize
        self.size = cameraSize[0] * cameraSize[1] * cameraSize[2]
        self.pipe = sp.Popen([FFMPEG_BIN, "-i", 'https://hls.ibb.gov.tr/' + cameraPath,
                              "-loglevel", "quiet",
                              "-an",
                              "-f", "image2pipe",
                              "-pix_fmt", "bgr24",
                              "-vcodec", "rawvideo", "-"],
                             stdin=sp.PIPE, stdout=sp.PIPE)

        raw_image = self.pipe.stdout.read(self.size)
        img0 = np.frombuffer(raw_image, dtype='uint8').reshape(self.cameraSize)
        self.event = click(img0, "alihan.txt", saveConfig=True)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        raw_image = self.pipe.stdout.read(self.size)
        img0 = np.frombuffer(raw_image, dtype='uint8').reshape(self.cameraSize)
        img1p = cv2.bitwise_and(img0, img0, mask=self.event.mask)
        img = cv2.add(img1p, self.event.vis)
        # img = letterbox(img1, self.img_size, stride=self.stride)[0]
        cv2.addWeighted(img1p, 0.5, img0, 0.5, 0, img0)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return 'cctv', img, img0, None

    def __len__(self):
        return 0
