from DataPoint import DataPoint
import cv2


class VideoOverlay:
  def __init__(self):
    self.shouldDrawBoxes = False
    self.shouldDrawLabels = False

    self.boundingBoxColor = (0,255,0)
    self.boundingBoxThickness = 2
    self.labelColor = (255,0,0)

  def setDrawLabels(self, shouldDrawLabels: bool):
    self.shouldDrawLabels = shouldDrawLabels

  def setDrawBoxes(self, shouldDrawBoxes: bool):
    self.shouldDrawBoxes = shouldDrawBoxes

  def processFrame(self, frame, frameIndex, dataPoint:DataPoint):
    if self.shouldDrawBoxes:
      bboxes = dataPoint.boundingBoxes
      if len(bboxes) <= frameIndex:
        return frame
      for x1,y1,x2,y2 in bboxes[frameIndex]:
        cv2.rectangle(frame,
          (x1,y1), (x2,y2),
          self.boundingBoxColor,
          self.boundingBoxThickness)

    if self.shouldDrawLabels:
      cv2.putText(frame,
        'Moth Detected',
        (x+w+10,y+h),
        0,
        1.0, # thickness
        self.labelColor,
        2)

    return frame
