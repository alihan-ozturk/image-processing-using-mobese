import cv2 as cv
import numpy as np

font = cv.FONT_HERSHEY_SIMPLEX


def click_event(event, x, y, flags, params):
    global img, copy, pts

    if event == cv.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv.putText(img, str(x) + ',' +
                   str(y), (x, y), font,
                   1, (255, 0, 0), 2)
        cv.imshow('img', img)
    elif event == cv.EVENT_RBUTTONDOWN and len(pts) > 0:
        copy = backup.copy()
        del pts[-1]
        for x, y in pts:
            cv.putText(copy, str(x) + ',' +
                       str(y), (x, y), font,
                       1, (255, 0, 0), 2)
        img = copy
        cv.imshow('img', img)


img = cv.imread("BESIKTAS SEHITLER TEPESI2023-03-24 01-09-43.png")
frameGray = np.zeros_like(img[:,:,0])
backup = img.copy()
pts = []
cv.imshow("img", img)
cv.setMouseCallback('img', click_event)
cv.waitKey(0)
cv.fillConvexPoly(frameGray, np.array(pts), (255, 255, 255))
cv.imshow("img", frameGray)
cv.waitKey(0)
frame_gray = cv.bitwise_and(backup, backup, mask=frameGray)
cv.imshow("img", frame_gray)
cv.waitKey(0)
