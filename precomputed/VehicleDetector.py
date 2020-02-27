# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3

import pickle
import numpy as np


class VehicleDetector:
  wantsRGB = True

  def __init__(self):
    self.frameNumber = 0

  def getFeatures(self, frame):
    boxes = self.boxes[self.frameNumber]
    scores = self.scores[self.frameNumber]

    # # Test what happens if we only run the detector every other frame
    # boxes = self.boxes[self.frameNumber - (self.frameNumber % 2)]
    # scores = self.scores[self.frameNumber - (self.frameNumber % 2)]

    self.frameNumber = (self.frameNumber + 1) % len(self.boxes)
    return boxes, scores

  def loadFeaturesFromDisk(self, featuresPath):
    self.frameNumber = 0
    with open(featuresPath, 'rb') as file:
      self.boxes, self.scores, _, _ = pickle.load(file)
