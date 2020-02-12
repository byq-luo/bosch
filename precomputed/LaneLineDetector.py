import pickle

class LaneLineDetector:
  def __init__(self):
    self.frameNumber = 0

  def getLines(self, frame):
    lanelines = self.lanelines[self.frameNumber]
    self.frameNumber = (self.frameNumber + 1) % len(self.lanelines)
    return lanelines

  def loadFeaturesFromDisk(self, featuresPath):
      assert (self.frameNumber == 0)
      with open(featuresPath, 'rb') as file:
          bboxes, segmentations, lines = pickle.load(file)
          self.lanelines = lines
