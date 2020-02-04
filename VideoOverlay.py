from DataPoint import DataPoint


class VideoOverlay:
  def __init__(self):
    self.shouldDrawBoxes = False
    self.shouldDrawLabels = False

    # TODO load overlay data from disk (or it's in dataPoint)
    self.frameNumberToData = {}

  # TODO connect Checkbox signals to this
  def setDrawLabels(self, shouldDrawLabels: bool):
    self.shouldDrawLabels = shouldDrawLabels

  # TODO connect Checkbox signals to this
  def setDrawBoxes(self, shouldDrawBoxes: bool):
    self.shouldDrawBoxes = shouldDrawBoxes


  def processFrame(self, qp):
    if self.shouldDrawBoxes:
      qp.drawRect(150,200, 40, 40)

    if self.shouldDrawLabels:
      qp.drawText(150, 190, "objTurnOff")


