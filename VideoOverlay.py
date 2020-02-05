from DataPoint import DataPoint
import cv2


class VideoOverlay:
  def __init__(self):
    self.shouldDrawBoxes = False
    self.shouldDrawLabels = False
    self.shouldDrawLaneLines = False

    self.boundingBoxColor = (0,255,0)
    self.boundingBoxThickness = 2
    self.labelColor = (255,0,0)

  def setDrawLabels(self, shouldDrawLabels: bool):
    self.shouldDrawLabels = shouldDrawLabels

  def setDrawBoxes(self, shouldDrawBoxes: bool):
    self.shouldDrawBoxes = shouldDrawBoxes
  
  def setDrawLaneLines(self, shouldDrawLaneLines: bool):
    self.shouldDrawLaneLines = shouldDrawLaneLines

  def processFrame(self, frame):
    x,y = 25, 25
    w,h = 100,100

    if self.shouldDrawBoxes:
      cv2.rectangle(frame,
        (x,y),
        (x+w,y+h),
        self.boundingBoxColor,
        self.boundingBoxThickness)

    if self.shouldDrawLabels:
      cv2.putText(frame,
        'Moth Detected',
        (x+w+10,y+h),
        0,
        1.0,
        self.labelColor,
        2)

    if self.shouldDrawLaneLines:
      pass

    return frame