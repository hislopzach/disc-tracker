# Disk Tracker

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

2. Press spacebar to move through video frame by frame, until the disk just leaves the throwers hand
3. Press 'a' key, then use your mouse to draw a rectangle around the disk
4. Press spacebar again, and watch as your throw is traced onto the video

## Limitations

Disk Tracker only works on static videos taken with a tripod (no panning or zooming).
This software may be lose track of the disk if the background is bright (i.e. the disk's path crosses the sun) or is similar in color to the disk (blue disk against blue sky).
The disk may not be tracked after hitting the ground

### credit

The project was developed for CPE 428 Intro to Computer Vision at Cal Poly, San Luis Obispo by Zachary Hislop, Lance Litten, and Nick Schnorr
