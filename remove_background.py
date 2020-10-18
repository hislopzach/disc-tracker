import cv2 as cv
from sys import argv

if len(argv) < 2:
    print("Usage: python remove_background <video> <background_image>")
    exit(2)
background = cv.imread(argv[2], cv.IMREAD_GRAYSCALE)

cap = cv.VideoCapture(argv[1])

videoWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(5)

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# size = (frame_width, frame_height)

fourcc = cv.VideoWriter_fourcc(*"MJPG")
# out = cv.VideoWriter("masked_video.avi", fourcc, 10, size)
out = cv.VideoWriter("output.avi", fourcc, frame_rate, (videoWidth, videoHeight), 1)
# fourcc = cv.VideoWriter_fourcc(*"XVID")
# out = cv.VideoWriter("output.avi", fourcc, 19.0, (320, 240))
if not cap.isOpened():
    print("Cannot open camera")
    exit()

totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
for i in range(int(totalFrames)):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype("uint8")

        abs_diff = cv.absdiff(gray, background).astype("uint8")

        ret2, otsu_thresh = cv.threshold(
            abs_diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        result_frame = cv.rotate(otsu_thresh, cv.ROTATE_180)
        resized = cv.resize(result_frame, (videoWidth, videoHeight), cv.INTER_LANCZOS4)
        color_converted = cv.cvtColor(resized, cv.COLOR_GRAY2RGB).astype("uint8")
        out.write(color_converted)
        cv.imshow("result", color_converted)
        cv.waitKey(0)

cap.release()
out.release()
cv.destroyAllWindows()
