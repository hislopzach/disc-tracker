import cv2 as cv
import numpy as np


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


def showDistance(bbox, cleaned_mask, frame):
    lowerb = int(bbox[0])
    upperb = int(bbox[0] + bbox[2])
    leftb = int(bbox[1])
    rightb = int(bbox[1] + bbox[3])
    cleaned_mask = np.where(cleaned_mask > 254, 1, 0)
    total = int(np.sum(cleaned_mask[lowerb:upperb, leftb:rightb]))
    dist = -6 * total + 1400
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(
        frame, str(dist), (lowerb, leftb), font, 1, (255, 255, 255), 1, cv.LINE_AA
    )
    return frame