class VideoOverlay:
  def __init__(self, videoPath: str):
    self.shouldDrawBoxes = False
    self.shouldDrawLabels = False

    # TODO load overlay data from disk
    self.frameNumberToData = {}

  # TODO connect Checkbox signals to this
  def setDrawBoxes(self, shouldDrawLabels: bool):
    self.shouldDrawLabels = shouldDrawLabels

  # TODO connect Checkbox signals to this
  def setDrawBoxes(self, shouldDrawBoxes: bool):
    self.shouldDrawBoxes = shouldDrawBoxes

  def processFrame(frame, frameNumber: int):
    return frame
