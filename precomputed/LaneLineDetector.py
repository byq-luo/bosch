# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3

import pickle
import numpy as np


class LaneLineDetector:
  def __init__(self):
    self.frameNumber = 0

  def getLines(self, frame):
    SMT = .9
    lines = self.lines[self.frameNumber]
    smooth_lines = []
    if len(lines) != 0:
      for coords,laneID in lines:
        smooth_coords = []
        px = None
        py = None
        for (x1, y1, x2, y2) in reversed(coords):
          if px is None:
            px = x1
            py = y1

          px = px * SMT + (1-SMT) * x1
          x1 = int(px)
          px = px * SMT + (1-SMT) * x2
          x2 = int(px)

          py = py * SMT + (1-SMT) * y1
          y1 = int(py)
          py = py * SMT + (1-SMT) * y2
          y2 = int(py)
          smooth_coords.append((x1, y1, x2, y2))
        smooth_lines.append((smooth_coords,laneID))
    self.frameNumber = (self.frameNumber + 1) % len(self.lines)
    return smooth_lines

  def loadFeaturesFromDisk(self, featuresPath):
    self.frameNumber = 0
    with open(featuresPath, 'rb') as file:
      _, _, self.lines, _ = pickle.load(file)
