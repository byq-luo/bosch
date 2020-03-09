import pickle

class LaneLineDetector:
  def __init__(self):
    self.frameIndex = 0

  def getLines(self, frame):
    lines = self.lines[self.frameIndex % len(self.lines)]
    self.frameIndex += 1
    return lines

  def loadFeaturesFromDisk(self, featuresPath):
    self.frameIndex  = 0
    with open(featuresPath, 'rb') as file:
      _, _, self.lines, _ = pickle.load(file)
