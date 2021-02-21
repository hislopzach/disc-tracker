# Disc Tracker

Make your own Follow Flights!

Requirements:
video taken with tripod
static video (no zoom or pan)

### Installation

1. Clone repo
2. Create virtual environment and install dependencies

```
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt
```

### Usage:

1. Run program with desired video as command line argument

```
$ python main.py video_name.mp4
```

2. Press spacebar to move through video frame by frame, until the disc just leaves the throwers hand
3. Press 'a' key, then use your mouse to draw a rectangle around the disc. Try to center the box around the center of the disc. If you are not satisifes with the location of the box, just click and drag to create a new one.
4. Press enter to confirm your bounding box, then watch as your throw is traced onto the video

## Limitations

Disc Tracker only works on static videos taken with a tripod (no panning or zooming).
This software may be lose track of the disc if the background is bright (i.e. the disc's path crosses the sun) or is similar in color to the disc (blue disc against blue sky).
The disc may not be tracked after hitting the ground

### credit

The project was developed for CPE 428 Intro to Computer Vision at Cal Poly, San Luis Obispo by Zachary Hislop, Lance Litten, and Nick Schnorr
