import cv2 as cv
import numpy as np
from sys import argv
# given a video, will create an image file containing the 'average frame'

cap = cv.VideoCapture(argv[1])
totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
print(totalFrames)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
frames = []
for i in range(int(totalFrames)):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)

avg_img = np.mean(frames, axis=0).astype("uint8")
cv.imshow("avg", avg_img)
cv.waitKey(0)
filename = argv[1][:-4] + "_averaged.png"
cv.imwrite(filename, avg_img)
