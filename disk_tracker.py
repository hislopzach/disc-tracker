import cv2 as cv
import collections
from pathlib import Path
from image_utils import *


class DiskTracker:

    WINDOW_NAME = "Result"

    def __init__(self, filename):
        self.filename = filename

        # set up video capture
        self.cap = cv.VideoCapture(filename)
        videoWidth = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_rate = self.cap.get(cv.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

        # setup video writer
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        output_filename = f"outputs/{Path(self.filename).stem}.avi"
        self.out_video = cv.VideoWriter(
            output_filename, fourcc, frame_rate, (videoWidth, videoHeight), 1
        )

        # setup tracker and background subtractor
        self.pts = collections.deque(maxlen=10000)
        self.tracker = cv.TrackerCSRT_create()
        self.back_sub = cv.createBackgroundSubtractorMOG2()

        self.overlay_started = False

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading video")
            exit()
        return frame

    def show_frame(self, frame):
        cv.imshow(self.WINDOW_NAME, frame)

    def optional_wait(self):
        cv.waitKey(1)

    def save_frame(self, frame):
        self.out_video.write(frame)

    def start_overlay(self, frame):
        self.overlay_started = True
        # get ROI from user
        bbox = cv.selectROI("Tracker", frame)
        inverted_frame = cv.bitwise_not(frame)
        # init background subtraction and tracking
        fg_mask = self.back_sub.apply(inverted_frame)
        cleaned_mask = clean_mask(fg_mask)
        self.tracker.init(cleaned_mask, bbox)
        cv.destroyWindow("Tracker")

        result_frame = self.process_frame(frame)
        return result_frame

    def process_frame(self, frame):
        if not self.overlay_started:
            return frame
        frame = frame.copy()
        inverted_frame = cv.bitwise_not(frame)
        fg_mask = self.back_sub.apply(inverted_frame)
        cleaned_mask = clean_mask(fg_mask)

        ok, bbox = self.tracker.update(cleaned_mask)
        if ok:
            self.pts.appendleft(
                (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            )
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # frame = showDistance(bbox, cleaned_mask, frame)
            cv.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

        for x in range(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[x - 1] is None or self.pts[x] is None:
                continue
            # otherwise, draw the connecting lines
            thickness = 3
            cv.line(frame, self.pts[x - 1], self.pts[x], (0, 0, 255), thickness)

        return frame
