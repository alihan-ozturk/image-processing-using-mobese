import time
import cv2
import numpy as np
import urllib.request

x = 5
y = 1200
links = []
for i in range(x):
    for j in range(y):
        print(i, j)
        newLink = f"https://hls.ibb.gov.tr/tkm{i}/hls/{j}.stream/chunklist.m3u8"
        try:
            urllib.request.urlretrieve(newLink)
            links.append(newLink)
        except urllib.error.HTTPError:
            continue
        

for i in range(100):
    for link in links:
        cap = cv2.VideoCapture(link)
        for _ in range(10):
            ret, frame = cap.read()
            if frame is None:
                time.sleep(np.exp(-1 * (10 - i)))
                continue
            else:
                name = link.replace("https://hls.ibb.gov.tr/", "")
                name = name.replace(".stream/chunklist.m3u8", "")
                name = name.replace("/", "")
                print(name, i)
                cv2.imwrite(name + str(i) + ".png", frame)
                break

