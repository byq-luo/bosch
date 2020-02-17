# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3

import pickle
import numpy as np


class VehicleDetector:
  wantsRGB = True

  def __init__(self):
    self.frameNumber = 0

  # Sometimes YOLO detects the host vehicle's dash. This function removes that detected bounding box.
  def removeBadBox(self, boxes, screenWidth):
    ret = []
    for box in boxes:
      width = (box[2] - box[0])
      if width / screenWidth < .8:
        ret.append(box)
    return ret

  # def removeCarsNotFacingAway():

  def getFeatures(self, frame):
    boxes = self.bboxes[self.frameNumber]
    scores = self.scores[self.frameNumber]
    segmentations = []

    # Filter boxes
    # return self.removeBadBox(bboxes, frame.shape[1]), segmentations
    # TODO !!!! give actual frame width
    boxes = self.removeBadBox(boxes, 700)

    self.frameNumber = (self.frameNumber + 1) % len(self.bboxes)
    return boxes, scores, segmentations

  def loadFeaturesFromDisk(self, featuresPath):
    assert(self.frameNumber == 0)
    with open(featuresPath, 'rb') as file:
      self.bboxes, self.scores, _, _ = pickle.load(file)
