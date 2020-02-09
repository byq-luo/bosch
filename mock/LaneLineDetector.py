import pickle

class LaneLineDetector:
  def __init__(self):
    with open('mock/videoData.pkl', 'rb') as file:
        bboxes, lanelines = pickle.load(file)
        self.lanelines = lanelines
    self.frameNumber = 0
  def getLines(self, frame):
    lanelines = self.lanelines[self.frameNumber % len(self.lanelines)]
    self.frameNumber += 1
    return lanelines