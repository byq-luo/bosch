# Fake video object that always loads a certain video

import sys
sys.path.append("..")
from Video import Video

import cv2

class Video(Video):
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture('mock/video.avi')
        if not self.vid.isOpened():
            raise ValueError("Unable to open mock video mock/video.avi")

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

        # https://www.pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
        self.numFrames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))