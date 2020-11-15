import cv2 as cv
import numpy as np
import sys
import collections

def check_boundaries(value, tolerance, ranges, upper_or_lower):
    if ranges == 0:
        # set the boundary for hue
        boundary = 180
    elif ranges == 1:
        # set the boundary for saturation and value
        boundary = 255

    if(value + tolerance > boundary):
        value = boundary
    elif (value - tolerance < 0):
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + tolerance
        else:
            value = value - tolerance
    return value

def pick_color(event,x,y,flags,param):
    global upper,lower, color_selected 
    if ((event == cv.EVENT_LBUTTONDOWN) & (color_selected==0)):
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        # Set range = 0 for hue and range = 1 for saturation and brightness
        # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
        hue_upper = check_boundaries(pixel[0], 10, 0, 1)
        hue_lower = check_boundaries(pixel[0], 10, 0, 0)
        saturation_upper = check_boundaries(pixel[1], 10, 1, 1)
        saturation_lower = check_boundaries(pixel[1], 10, 1, 0)
        value_upper = check_boundaries(pixel[2], 10, 1, 1)
        value_lower = check_boundaries(pixel[2], 10, 1, 0)
        upper =  np.array([hue_upper, saturation_upper, value_upper])
        lower =  np.array([hue_lower, saturation_lower, value_lower])
        print(lower, upper)
        image_mask = cv.inRange(image_hsv,lower,upper)
        cv.imshow("Mask",image_mask)
        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 


def main():
    print("Usage:color_picker_tracker video_filenameaa")
    global image_hsv, pixel,totalFrames, upper, lower, color_selected
    video_filename = sys.argv[1]
    cap = cv.VideoCapture(video_filename)
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)
    color_selected=0;
    pts = collections.deque(maxlen=64)
    summed_img = np.zeros((videoHeight, videoWidth, 3)).astype("uint8")
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    tracker = cv.TrackerCSRT_create()
    first_frame = 1
    roiselected=-1
    ret, frame = cap.read()
    for i in range(int(totalFrames)):
        # Capture frame-by-frame

        ret, frame = cap.read()
        if ((ret) and (color_selected==1)):
            frame_HSV = cv.cvtColor(cv.bitwise_not(frame), cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, lower, upper)
        #frame_threshold2 = cv.inRange(frame_HSV, (170, 70, 50), (180, 255, 255))
        #frame_threshold = frame_threshold|frame_threshold2 
        #frame_threshold = cv.erode(frame_threshold,None,iterations=1)
        #frame_threshold = cv.dilate(frame_threshold,None,iterations=1)
        if (roiselected==-1):
            cv.imshow('Tracker', frame)
            key=cv.waitKey()
            if key == ord('a') or key == 27:
                frame=cv.bitwise_not(frame)
                image_hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                cv.imshow("HSV",image_hsv)
                cv.setMouseCallback("HSV", pick_color)
                while (color_selected==0):
                    key=cv.waitKey(0)
                    if key == ord('a'):
                        color_selected=1
                        cv.destroyAllWindows()
                frame_threshold = cv.inRange(image_hsv, lower, upper)
                bbox = cv.selectROI('Tracker', frame)
                tracker.init(frame_threshold, bbox)
                roiselected=1
                cv.destroyAllWindows()
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