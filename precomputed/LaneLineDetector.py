# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3

import pickle
import numpy as np


class LaneLineDetector:
  def __init__(self):
    self.frameNumber = 0

  def getLines(self, frame):
    lines = self.lines[self.frameNumber]
    self.frameNumber = (self.frameNumber + 1) % len(self.lines)
    return lines

  def loadFeaturesFromDisk(self, featuresPath):
    self.frameNumber = 0
    with open(featuresPath, 'rb') as file:
      _, _, _, self.lines, _ = pickle.load(file)
