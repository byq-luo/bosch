from DataPoint import DataPoint
import cv2


class VideoOverlay:
  def __init__(self):
    self.shouldDrawBoxes = False
    self.shouldDrawLabels = False
    self.shouldDrawLaneLines = False
    self.shouldDrawSegmentations = False

    self.segmentationsColor = (255, 0, 0)
    self.boundingBoxColor = (0,255,0)
    self.boundingBoxThickness = 2
    self.labelColor = (255,0,0)

  def setDrawLabels(self, shouldDrawLabels: bool):
    self.shouldDrawLabels = shouldDrawLabels

  def setDrawBoxes(self, shouldDrawBoxes: bool):
    self.shouldDrawBoxes = shouldDrawBoxes

  def setDrawLaneLines(self, shouldDrawLaneLines: bool):
    self.shouldDrawLaneLines = shouldDrawLaneLines

  def setDrawSegmentations(self, shouldDrawSegmentations: bool):
    self.shouldDrawSegmentations = shouldDrawSegmentations

  def processFrame(self, frame, frameIndex, dataPoint:DataPoint, currentTime):

    # TODO
    x = 30
    y = 45

    if self.shouldDrawBoxes:
      bboxes = dataPoint.boundingBoxes
      if len(bboxes) <= frameIndex:
        return frame
      for x1,y1,x2,y2 in bboxes[frameIndex]:
        x, y = x1, y1-15
        cv2.rectangle(frame,
          (x1,y1), (x2,y2),
          self.boundingBoxColor,
          self.boundingBoxThickness)

    if self.shouldDrawLabels:
      currentLabel = "No Label Yet"
      for line in dataPoint.predictedLabels:
        if float(line[1]) < currentTime:
          currentLabel = line[0]
        else:
          break

      cv2.putText(frame,
                  currentLabel,
                  (x,y),
                  0, 1,
                  self.labelColor)

    if self.shouldDrawSegmentations:
      if frameIndex < len(dataPoint.segmentations):
        for boundary in dataPoint.segmentations[frameIndex]:
          cv2.drawContours(frame, boundary, 0, self.segmentationsColor, 2)

    if self.shouldDrawLaneLines:
      if len(dataPoint.laneLines) <= frameIndex:
        return frame
      lines = dataPoint.laneLines[frameIndex]
      for x1, y1, x2, y2 in lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame
