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
                result_frame, summed_mask = highlight_movement(frame, background_image, summed_mask)
            else:
                result_frame = frame
            cv.imshow("result frame", result_frame)
            cv.waitKey(5)


def show_knn_removed_background(video_filename, save=False):
    cap = cv.VideoCapture(video_filename)
    videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(5)
    # setup video writer
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter("output.avi", fourcc, frame_rate, (videoWidth, videoHeight), 1)

    back_sub = cv.createBackgroundSubtractorKNN()
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
            fgMask = back_sub.apply(frame)
            if start_overlay:
                # eroded_mask = erode_and_dilate(fgMask, (5, 5))
                # cleaned_mask = remove_small_components(fgMask, 20)
                # cleaned_mask = remove_large_components(cleaned_mask, 1200)
                cleaned_mask = keep_medium_components(fgMask, 20, 1200)
                # cleaned_mask = erode_and_dilate(cleaned_mask, (5, 5))
                summed_mask += cleaned_mask
                result_frame = add_overlay(frame, summed_mask)
            else:
                result_frame = frame
            cv.imshow("result frame", result_frame)
            cv.waitKey(5)
            if save:
                out.write(result_frame)
    cap.release()
    out.release()


def remove_lone_pixels(image):
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


def keep_medium_components(image, min_size=25, max_size=500):
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
        if sizes[i] <= max_size or sizes[i] >= min_size:
            result[output == i + 1] = 255

    return result.astype("uint8")


def highlight_movement(frame, background_image, summed_mask):
    video_width = frame.shape[1]
    video_height = frame.shape[0]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype("uint8")
    abs_diff = cv.absdiff(gray, background_image).astype("uint8")
    ret2, otsu_thresh = cv.threshold(abs_diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = otsu_thresh
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(otsu_thresh, kernel, iterations=1)
    mask = cv.dilate(otsu_thresh, kernel, iterations=1)

    resized_mask = cv.resize(mask, (video_width, video_height), cv.INTER_LANCZOS4)
    summed_mask += resized_mask

    color_converted = cv.cvtColor(resized_mask, cv.COLOR_GRAY2RGB).astype("uint8")
    result_frame = add_overlay(frame, summed_mask, 0.8)
    return result_frame, summed_mask


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
    print(mask)
    rgba[:, :, 3] = mask
    return rgba


def erode_and_dilate(img, kernel=(3, 3)):
    img = img.copy()
    img = cv.erode(img, kernel)
    img = cv.dilate(img, kernel)
    return img


def main():
    video_filename = sys.argv[1]
    # background_image = get_background_image(video_filename)

    # show_removed_background(video_filename, background_image)
    prompt = input("save video?")
    save_video = "y" in prompt
    show_knn_removed_background(video_filename, save_video)


if __name__ == "__main__":
    main()