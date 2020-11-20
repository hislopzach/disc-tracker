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
    summed_mask = np.zeros((videoHeight, videoWidth)).astype("uint8")
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    start_overlay = False
    for i in range(int(totalFrames)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not start_overlay:
            cv.imshow("frame", frame)
            key = cv.waitKey()
            if key == ord("a"):
                start_overlay = True
        if ret:
            if start_overlay:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype("uint8")
                abs_diff = cv.absdiff(gray, background_image).astype("uint8")
                ret2, otsu_thresh = cv.threshold(abs_diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                mask = otsu_thresh
                kernel = np.ones((3, 3), np.uint8)
                mask = cv.erode(otsu_thresh, kernel, iterations=1)
                mask = cv.dilate(otsu_thresh, kernel, iterations=1)

                resized_mask = cv.resize(mask, (videoWidth, videoHeight), cv.INTER_LANCZOS4)
                summed_mask += resized_mask

                color_converted = cv.cvtColor(resized_mask, cv.COLOR_GRAY2RGB).astype("uint8")
                summed_img += color_converted
                result_frame = add_overlay(frame, summed_mask, 0.8)
            else:
                result_frame = frame
            cv.imshow("result frame", result_frame)
            cv.waitKey(5)


def add_overlay(frame, mask, alpha):
    colored_overlay = np.zeros(frame.shape, frame.dtype)
    colored_overlay[:, :] = (0, 0, 255)
    colored_overlay = cv.bitwise_and(colored_overlay, colored_overlay, mask=mask)
    res = frame.copy()
    beta = 1 - alpha
    cv.addWeighted(colored_overlay, alpha, res, beta, 0, res)
    return res


def main():
    video_filename = sys.argv[1]
    background_image = get_background_image(video_filename)

    show_removed_background(video_filename, background_image)


if __name__ == "__main__":
    main()