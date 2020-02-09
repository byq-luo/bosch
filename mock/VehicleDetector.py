# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3
from mock.Video import Video
import pickle

class VehicleDetector:
    wantsRGB = True
    def __init__(self):
        with open('mock/videoData.pkl', 'rb') as file:
            bboxes, _, segmentations = pickle.load(file)
            self.bboxes = bboxes
            self.segmentations = segmentations
        self.frameNumber = 0

    def getBoxes(self, frame):
        bboxes = self.bboxes[self.frameNumber % len(self.bboxes)]
        segmentations = self.segmentations[self.frameNumber % len(self.segmentations)]
        self.frameNumber += 1
        return bboxes, segmentations