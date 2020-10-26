import cv2 as cv
import numpy as np
from sys import argv

"""given a video and its average image, creates an image (traced.png) and a video (output.avi)"""
if len(argv) < 2:
    print("Usage: python remove_background <video> <background_image>")
    exit(2)
background = cv.imread(argv[2], cv.IMREAD_GRAYSCALE)

cap = cv.VideoCapture(argv[1])

videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(5)

fourcc = cv.VideoWriter_fourcc(*"MJPG")
out = cv.VideoWriter("output.avi", fourcc, frame_rate, (videoWidth, videoHeight), 1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

summed_img = np.zeros((videoHeight, videoWidth, 3)).astype("uint8")
totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
for i in range(int(totalFrames)):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype("uint8")
        abs_diff = cv.absdiff(gray, background).astype("uint8")
        ret2, otsu_thresh = cv.threshold(
            abs_diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        result_frame = otsu_thresh

        result_frame = cv.rotate(otsu_thresh, cv.ROTATE_180)

        resized = cv.resize(result_frame, (videoWidth, videoHeight), cv.INTER_LANCZOS4)
        color_converted = cv.cvtColor(resized, cv.COLOR_GRAY2RGB).astype("uint8")
        summed_img += color_converted
        out.write(summed_img)
        cv.imshow("summed", summed_img)
        cv.waitKey(1)

cv.imwrite("traced.png", summed_img)

cap.release()
out.release()
cv.destroyAllWindows()
