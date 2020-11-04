import cv2 as cv
import numpy as np
import sys


def get_background_image(video_filename):
    cap = cv.VideoCapture(video_filename)
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
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
    return avg_img


def show_removed_background(video_filename, background_image):
    cap = cv.VideoCapture(video_filename)
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)

    summed_img = np.zeros((videoHeight, videoWidth, 3)).astype("uint8")
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    for i in range(int(totalFrames)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype("uint8")
            abs_diff = cv.absdiff(gray, background_image).astype("uint8")
            ret2, otsu_thresh = cv.threshold(
                abs_diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )
            result_frame = otsu_thresh

            # result_frame = cv.rotate(otsu_thresh, cv.ROTATE_180)

            resized = cv.resize(
                result_frame, (videoWidth, videoHeight), cv.INTER_LANCZOS4
            )
            color_converted = cv.cvtColor(resized, cv.COLOR_GRAY2RGB).astype("uint8")
            summed_img += color_converted
            cv.imshow("summed", summed_img)
            # cv.imshow("current_frame", result_frame)
            cv.waitKey(5)


def main():
    video_filename = sys.argv[1]
    background_image = get_background_image(video_filename)

    show_removed_background(video_filename, background_image)


if __name__ == "__main__":
    main()