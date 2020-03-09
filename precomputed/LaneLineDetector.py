import pickle

class LaneLineDetector:
  def getLines(self, frame, frameIndex):
    return self.lines[frameIndex]

  def loadFeaturesFromDisk(self, featuresPath):
    with open(featuresPath, 'rb') as file:
      _, _, self.lines, _ = pickle.load(file)
