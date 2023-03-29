import cv2 as cv
import numpy as np
import json


class click:
    FONT = cv.FONT_HERSHEY_SIMPLEX
    ALPHA = 0.5
    KEY = ord("s")

    def __init__(self, img, configName="config.txt", saveConfig=False):
        self.__img = img.copy()
        self.__backup = img.copy()
        self.allPts = []
        try:
            with open(configName, 'r') as f:
                try:
                    self.allPts = json.load(f)
                    mask = np.zeros_like(self.__img[:, :, 0])
                    for pts in self.allPts:
                        cv.fillConvexPoly(mask, np.array(pts), 255)
                        masked = cv.bitwise_and(self.__backup, self.__backup, mask=mask)
                    cv.addWeighted(masked, self.ALPHA, self.__img, 1 - self.ALPHA, 0, self.__img)
                except json.decoder.JSONDecodeError:
                    print("file doesnt contain pts")
        except FileNotFoundError:
            print("will be create " + configName + " file")
        self.__temp = self.__img.copy()
        self.__pts = []
        self.mask = None
        self.__createMask()
        if saveConfig:
            with open(configName, 'w') as f:
                json.dump(self.allPts, f)

    def __clickEvent(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            self.__pts.append([x, y])
            cv.putText(self.__img, str(len(self.__pts)), (x, y), self.FONT,
                       1, (255, 0, 0), 2)
            cv.imshow('img', self.__img)
        elif event == cv.EVENT_RBUTTONDOWN and len(self.__pts) > 0:
            copy = self.__temp.copy()
            del self.__pts[-1]
            for i, (x, y) in enumerate(self.__pts):
                cv.putText(copy, str(i + 1), (x, y), self.FONT,
                           1, (255, 0, 0), 2)
            self.__img = copy
            cv.imshow('img', self.__img)

    def __createMask(self):

        cv.imshow("img", self.__img)
        cv.setMouseCallback('img', self.__clickEvent)
        __KEY = cv.waitKey(0)

        if len(self.__pts) > 0:
            self.allPts.append(self.__pts)
            self.__pts = []
        if len(self.allPts) == 0:
            raise Exception("no pts")
        mask = np.zeros_like(self.__img[:, :, 0])
        for i in range(len(self.allPts)):
            cv.fillConvexPoly(mask, np.array(self.allPts[i]), 255)
            masked = cv.bitwise_and(self.__backup, self.__backup, mask=mask)
        if __KEY == self.KEY:
            self.__img = self.__backup.copy()
            cv.addWeighted(masked, self.ALPHA, self.__img, 1 - self.ALPHA, 0, self.__img)
            self.__temp = self.__img.copy()
            self.__createMask()
        else:
            cv.destroyWindow("img")
            del self.__img
            del self.__backup
            del self.__temp
            self.mask = mask

    def applyMask(self, img):
        masked = cv.bitwise_and(img, img, mask=self.mask)
        cv.addWeighted(masked, self.ALPHA, img, 1 - self.ALPHA, 0, img)
        return masked, img


img = cv.imread("img.png")
event = click(img,"alihan.txt", saveConfig=True)
print(event.allPts)
imgC, imgD = event.applyMask(img)
cv.imshow("test", imgD)
cv.imshow("test2", imgC)
cv.waitKey(0)
