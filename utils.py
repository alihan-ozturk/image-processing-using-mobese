import cv2 as cv
import numpy as np

font = cv.FONT_HERSHEY_SIMPLEX


def click_event(event, x, y, flags, params):
    global img, copy, pts
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv.putText(img, str(len(pts)), (x, y), font,
                   1, (255, 0, 0), 2)
        cv.imshow('img', img)
    elif event == cv.EVENT_RBUTTONDOWN and len(pts) > 0:
        copy = backup.copy()
        del pts[-1]
        for i, (x, y) in enumerate(pts):
            cv.putText(copy, str(i+1), (x, y), font,
                       1, (255, 0, 0), 2)
        img = copy
        cv.imshow('img', img)


img = cv.imread("BESIKTAS SEHITLER TEPESI2023-03-24 01-53-01.png")
mask = np.zeros_like(img[:, :, 0])
backup = img.copy()
pts = []
cv.imshow("img", img)
cv.setMouseCallback('img', click_event)
cv.waitKey(0)
cv.fillConvexPoly(mask, np.array(pts), 255)
print(mask.shape)
cv.imshow("img", mask)
cv.waitKey(0)
frame_gray = cv.bitwise_and(backup, backup, mask=mask)
cv.imshow("img", frame_gray)
cv.waitKey(0)
print(pts)
