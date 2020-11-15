import cv2 as cv
import numpy as np
import sys
import collections

def main():
    #video_filename = sys.argv[1]
    cap = cv.VideoCapture("xcaliber_198.MOV")
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)
    pts = collections.deque(maxlen=64)
    summed_img = np.zeros((videoHeight, videoWidth, 3)).astype("uint8")
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    tracker = cv.TrackerCSRT_create()
    first_frame = 1
    roiselected=-1
    for i in range(int(totalFrames)):
        # Capture frame-by-frame

        ret, frame = cap.read()
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (0, 70, 50), (10, 255, 255))
        frame_threshold2 = cv.inRange(frame_HSV, (170, 70, 50), (180, 255, 255))
        frame_threshold = frame_threshold|frame_threshold2 
        #frame_threshold = cv.erode(frame_threshold,None,iterations=1)
        #frame_threshold = cv.dilate(frame_threshold,None,iterations=1)
        if (roiselected==-1):
            cv.imshow('Tracker', frame)
            key=cv.waitKey()
            if key == ord('a') or key == 27:
                bbox = cv.selectROI('Tracker', frame)
                tracker.init(frame_threshold, bbox)
                roiselected=1
        else:
            if ret:
                ok, bbox = tracker.update(frame_threshold)
                if ok:
                    pts.appendleft((int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)))
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                for x in range(1, len(pts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if pts[x - 1] is None or pts[x] is None:
                        continue
                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = 3
                    cv.line(frame, pts[x - 1], pts[x], (0, 0, 255), thickness)
                cv.imshow('Tracker', frame)
                cv.waitKey(5)
    
    #cv.imshow('FRAME', frame)
    cv.waitKey()
if __name__ == "__main__":
    main()
cv.destroyAllWindows()