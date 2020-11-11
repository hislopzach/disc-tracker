import cv2 as cv
import numpy as np
import sys


def main():
    video_filename = sys.argv[1]
    cap = cv.VideoCapture(video_filename)
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)

    summed_img = np.zeros((videoHeight, videoWidth, 3)).astype("uint8")
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    tracker = cv.TrackerMIL_create()

    first_frame = 1

    for i in range(int(totalFrames)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if first_frame:
            bbox = cv.selectROI('Tracker', frame)
            tracker.init(frame, bbox)
            first_frame = 0
        if ret:
            ok, bbox = tracker.update(frame)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv.imshow('Tracker', frame)
            cv.waitKey(5)


if __name__ == "__main__":
    main()