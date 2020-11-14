from __future__ import print_function
import cv2 as cv
import argparse
# tested with xcaliber_198.MOV
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
cap = cv.VideoCapture("xcaliber_198.MOV")
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

while True:
    
    ret, frame = cap.read()
    if frame is None:
        break
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (0, 70, 50), (10, 255, 255))
    frame_threshold2 = cv.inRange(frame_HSV, (170, 70, 50), (180, 255, 255))
    #frame_threshold = cv.erode(frame_threshold,None,iterations=2)
    #frame_threshold = cv.dilate(frame_threshold,None,iterations=2)
    frame_threshold = frame_threshold|frame_threshold2 
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    
    key = cv.waitKey()
    if key == ord('q') or key == 27:
        break
cv.destroyAllWindows()