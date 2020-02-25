from VehicleTracker import Vehicle

def isLeft(a, b, c):
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;

class LabelGenerator:
  def __init__(self, videoFPS):
    self.labels

  def getLabels(self):
    return self.labels

  def processFrame(self, vehicles, lines, frameIndex):
    if len(lines) != 2:
      return


