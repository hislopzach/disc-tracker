import cv2 as cv
import sys
from disc_tracker import DiscTracker


def usage():
    print(f"Usage: python {__file__} <video path>")


def main():
    if len(sys.argv) < 2:
        usage()
        exit()
    video_filename = sys.argv[1]
    # print(
    #     "Usage: press space key to advance video. \nWhen disc has been released, press 'a' to select the disc in the ROI then press space begin tracking the flight of the disc"
    # )
    prompt = input("save video? (y/N): ")
    save_video = "y" in prompt
    dt = DiscTracker(video_filename)

    result_frame = None

    frame = dt.get_frame()
    dt.show_frame(frame)

    for _ in range(dt.frame_count - 1):
        if dt.overlay_started:
            frame = dt.get_frame()
            result_frame = dt.process_frame(frame)
        else:
            key = cv.waitKey()
            if key == ord("a"):
                dt.start_overlay(frame)
                result_frame = dt.process_frame(frame)
            else:
                frame = dt.get_frame()
                dt.update_background(frame)
                result_frame = frame

        dt.show_frame(result_frame)
        if save_video:
            dt.save_frame(result_frame)
    dt.cleanup()


if __name__ == "__main__":
    main()