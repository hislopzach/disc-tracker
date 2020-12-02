import cv2 as cv
import numpy as np
import sys
import collections
from pathlib import Path


def show_knn_removed_background(video_filename, save=False):
    tracker = cv.TrackerCSRT_create()
    cap = cv.VideoCapture(video_filename)
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)
    # setup video writer
    output_filename = f"outputs/{Path(video_filename).stem}.avi"
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(
        output_filename, fourcc, frame_rate, (videoWidth, videoHeight), 1
    )
    pts = collections.deque(maxlen=10000)
    back_sub = cv.createBackgroundSubtractorMOG2()
    summed_mask = np.zeros((videoHeight, videoWidth)).astype("uint8")
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    start_overlay = False
    for i in range(int(totalFrames)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        inverted_frame = cv.bitwise_not(frame)
        if not start_overlay:
            cv.imshow("frame", frame)
            key = cv.waitKey()
            if key == ord("a"):
                start_overlay = True
                bbox = cv.selectROI("Tracker", frame)
                fgMask = back_sub.apply(inverted_frame)
                cleaned_mask = clean_mask(fgMask)
                tracker.init(cleaned_mask, bbox)
                cv.destroyWindow("frame")
        if ret:
            fgMask = back_sub.apply(inverted_frame)
            if start_overlay:
                cleaned_mask = clean_mask(fgMask)

                ok, bbox = tracker.update(cleaned_mask)
                if ok:
                    pts.appendleft(
                        (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                    )
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    frame = showDistance(bbox, cleaned_mask, frame)
                    cv.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                result_frame = frame
            else:
                result_frame = frame
            cv.destroyWindow("Tracker")
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
            cv.imshow("result frame", result_frame)
            cv.waitKey(1)
            if save:
                out.write(result_frame)
    cap.release()
    out.release()

def showDistance(bbox, cleaned_mask, frame):
    lowerb = int(bbox[0])
    upperb = int(bbox[0] + bbox[2])
    leftb = int(bbox[1])
    rightb = int(bbox[1] + bbox[3])
    cleaned_mask = np.where(cleaned_mask > 254, 1, 0)
    total = int(np.sum(cleaned_mask[lowerb:upperb, leftb:rightb]))
    dist = -6 * total + 1400
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, str(dist), (lowerb, leftb), font, 1, (255, 255, 255), 1, cv.LINE_AA)
    return frame

def clean_mask(mask):
    cleaned_mask = remove_lone_pixels(mask)
    cleaned_mask = remove_small_components(cleaned_mask, 15)
    # cleaned_mask = remove_large_components(cleaned_mask, 1200)
    # cleaned_mask = cv.GaussianBlur(cleaned_mask, (3, 3), 0)
    cleaned_mask = erode_and_dilate(cleaned_mask, (5, 5))
    return cleaned_mask


def remove_lone_pixels(image):
    # uses a filter to remove lone pixels
    image = image.copy()
    image_comp = cv.bitwise_not(image)  # could just use 255-image

    kernel1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)

    hitormiss1 = cv.morphologyEx(image, cv.MORPH_ERODE, kernel1)
    hitormiss2 = cv.morphologyEx(image_comp, cv.MORPH_ERODE, kernel2)
    hitormiss = cv.bitwise_and(hitormiss1, hitormiss2)

    hitormiss_comp = cv.bitwise_not(hitormiss)  # could just use 255-img
    result = cv.bitwise_and(image, image, mask=hitormiss_comp)
    return result


def remove_small_components(image, min_size=25):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(
        image, connectivity=8
    )
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    result = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            result[output == i + 1] = 255

    return result.astype("uint8")


def remove_large_components(image, max_size=150):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(
        image, connectivity=8
    )
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:]
    nb_components = nb_components - 1

    # your answer image
    result = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i][-1] <= max_size and sizes[i][-2] <= 200:
            result[output == i + 1] = 255

    return result.astype("uint8")


def add_overlay(frame, mask, color=(0, 0, 255), alpha=0.8):
    colored_overlay = np.zeros(frame.shape, frame.dtype)
    colored_overlay[:, :] = color
    colored_overlay = cv.bitwise_and(colored_overlay, colored_overlay, mask=mask)
    # make mask transparent to avoid darkening image
    transparent_overlay = make_overlay_transparent(colored_overlay, mask)

    # convert to BRGA to add transparent overlay
    frame_alpha = cv.cvtColor(frame.copy(), cv.COLOR_BGR2BGRA)
    res = cv.add(frame_alpha, transparent_overlay)
    # convert back bgr
    res = cv.cvtColor(res, cv.COLOR_BGRA2BGR)
    return res


def make_overlay_transparent(colored_overlay, mask):
    rgba = cv.cvtColor(colored_overlay, cv.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


def erode_and_dilate(img, kernel=(3, 3)):
    img = img.copy()
    img = cv.erode(img, kernel)
    img = cv.dilate(img, kernel)
    return img


def main():
    video_filename = sys.argv[1]
    print(
        "Usage: press space key to advance video. \nWhen disc has been released, press 'a' to select the disc in the ROI then press space begin tracking the flight of the disc"
    )
    prompt = input("save video? (y/N)")
    save_video = "y" in prompt
    show_knn_removed_background(video_filename, save_video)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()