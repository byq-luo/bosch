# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3

import pickle

class VehicleDetector:
    wantsRGB = True
    def __init__(self):
        self.frameNumber = 0

    def getFeatures(self, frame):
        bboxes = self.bboxes[self.frameNumber]
        segmentations = self.segmentations[self.frameNumber % len(self.segmentations)]
        self.frameNumber = (self.frameNumber + 1) % len(self.bboxes)
        return bboxes, segmentations

    def loadFeaturesFromDisk(self, featuresPath):
        assert(self.frameNumber==0)
        with open(featuresPath, 'rb') as file:
            bboxes, segmentations, lines = pickle.load(file)
            self.bboxes = bboxes
            self.segmentations = segmentations