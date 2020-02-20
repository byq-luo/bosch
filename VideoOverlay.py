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
    self.boundingBoxThickness = 1
    self.labelColor = (255,0,0)
    self.laneColors = [(255,0,0),(0,255,0),(0,0,255),(255,191,0)]

  def setDrawLabels(self, shouldDrawLabels: bool):
    self.shouldDrawLabels = shouldDrawLabels

  def setDrawBoxes(self, shouldDrawBoxes: bool):
    self.shouldDrawBoxes = shouldDrawBoxes

  def setDrawLaneLines(self, shouldDrawLaneLines: bool):
    self.shouldDrawLaneLines = shouldDrawLaneLines

  def setDrawSegmentations(self, shouldDrawSegmentations: bool):
    self.shouldDrawSegmentations = shouldDrawSegmentations

  def processFrame(self, frame, frameIndex, dataPoint:DataPoint, currentTime):
    if self.shouldDrawBoxes:
      bboxes = dataPoint.boundingBoxes
      if len(bboxes) > frameIndex:
        for (bx1,by1,bx2,by2),(x1,y1,x2,y2),_id in bboxes[frameIndex]:
          x, y = x1,y1-7
          cv2.rectangle(frame,
            (x1,y1), (x2,y2),
            self.boundingBoxColor,
            self.boundingBoxThickness)
          cv2.rectangle(frame,
            (bx1,by1), (bx2,by2),
            (0,0,255),
            self.boundingBoxThickness)
          cv2.putText(frame,
                      str(_id),
                      (x,y),
                      0, .3,
                      self.labelColor)

    if self.shouldDrawLabels:
      currentLabel = "No Label Yet"
      for line in dataPoint.predictedLabels:
        if float(line[1]) < currentTime:
          currentLabel = line[0]
        else:
          break
      cv2.putText(frame,
                  currentLabel,
                  (30, 45),
                  0, 1,
                  self.labelColor)

    if self.shouldDrawSegmentations:
      if frameIndex < len(dataPoint.segmentations):
        for boundary in dataPoint.segmentations[frameIndex]:
          cv2.drawContours(frame, boundary, 0, self.segmentationsColor, 2)

    # if self.shouldDrawLaneLines:
    #   if len(dataPoint.laneLines) > frameIndex:
    #     lines = dataPoint.laneLines[frameIndex]
    #     if len(lines) == 0:
    #       return frame
    #     for coords,laneID in lines:
    #       for (x1, y1, x2, y2) in coords:
    #         cv2.line(frame, (x1, y1), (x2, y2), self.laneColors[laneID], 2)
    if self.shouldDrawLaneLines:
      if len(dataPoint.laneLines) > frameIndex:
        lines = dataPoint.laneLines[frameIndex]
        for (x1, y1, x2, y2),laneID in lines:
          cv2.line(frame, (x1, y1), (x2, y2), self.laneColors[laneID], 2)

    frame = cv2.copyMakeBorder(frame, 
                    3, 3, 3, 3, 
                    cv2.BORDER_CONSTANT, 
                    value=(0,0,0))

    return frame
