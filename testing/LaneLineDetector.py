import pickle

class LaneLineDetector:
  def __init__(self):
    self.frameNumber = 0

  def getLines(self, frame):
    lanelines = self.lanelines[self.frameNumber % len(self.lanelines)]
    self.frameNumber += 1
    return lanelines

  def loadFeaturesFromDisk(self, featuresPath):
      assert (self.frameNumber == 0)
      with open(featuresPath, 'rb') as file:
          bboxes, _, segmentations = pickle.load(file)
          self.bboxes = bboxes
          self.segmentations = segmentations
