import cv2

class Video:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

        # https://www.pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
        self.numFrames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def getFrame(self):
        isFrameAvailable = False
        if self.vid.isOpened():
            isFrameAvailable, frame = self.vid.read()
            if isFrameAvailable:
                return (isFrameAvailable, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (isFrameAvailable, None)
        else:
            return(isFrameAvailable, None)

    def getFps(self):
        return self.fps

    def getTotalNumFrames(self):
        return self.numFrames

    def getFrameNumber(self):
        return int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    def setFrameNumber(self, frameIndex):
        assert(frameIndex < self.numFrames)
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)

    def getVideoLength(self):
        return self.numFrames / self.fps

    def getCurrentTime(self):
        return self.vid.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
