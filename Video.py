import cv2

class Video:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))

        # https://www.pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
        self.num_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        isFrameAvailable = False
        if self.vid.isOpened():
            isFrameAvailable, frame = self.vid.read()
            if isFrameAvailable:
                return (isFrameAvailable, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (isFrameAvailable, None)
        else:
            return(isFrameAvailable, None)

    def get_fps(self):
        return self.fps

    def get_total_num_frames(self):
        return self.num_frames

    def get_frame_number(self):
        return int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    def set_frame_number(self, frame):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def getVideoLength(self):
        return self.num_frames / self.fps

    def getCurrentTime(self):
        return self.vid.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
