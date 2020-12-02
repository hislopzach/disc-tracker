import cv2 as cv
import numpy as np
import sys
import collections
from pathlib import Path

pts = []


def show_knn_removed_background(video_filename, save=False):
    cap = cv.VideoCapture(video_filename)
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)
    # setup video writer
    output_filename = f"outputs/{Path(video_filename).stem}_manual.avi"
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(
        output_filename, fourcc, frame_rate, (videoWidth, videoHeight), 1
    )

    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    start_overlay = False
    stop_overlay = False
    for i in range(int(totalFrames)):
        if stop_overlay:
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if not start_overlay:
                cv.imshow("frame", frame)
                key = cv.waitKey()
                if key == ord("a"):
                    start_overlay = True
                    cv.setMouseCallback("frame", click_event)
            if start_overlay:
                key = cv.waitKey()
                if key == ord("q"):
                    stop_overlay = True
                for x in range(1, len(pts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if pts[x - 1] is None or pts[x] is None:
                        continue
                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = 3
                    cv.line(frame, pts[x - 1], pts[x], (0, 0, 255), thickness)
                result_frame = frame
                cv.imshow("frame", result_frame)
            else:
                result_frame = frame
                cv.imshow("frame", frame)
            if save:
                out.write(result_frame)
    cap.release()
    out.release()


def click_event(event, x, y, flags, params):
    global pts
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        # print(pts)


def main():
    video_filename = sys.argv[1]
    print(
        "Usage: press space key to advance video. \nWhen disc has been released, press 'a' to select the disc in the ROI then press space begin tracking the flight of the disc"
    )
    prompt = input("save video? (y/N) ")
    save_video = "y" in prompt
    show_knn_removed_background(video_filename, save_video)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()