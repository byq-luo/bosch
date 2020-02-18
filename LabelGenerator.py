from VehicleTracker import Vehicle

class LabelGenerator:
  def __init__(self, videoFPS):
    self.currentTargetObject = None
    self.newPotentialTarget = None
    self.newEventTimer = 10
    self.newTargetTimer = 10
    self.lastLabelProduced = None
    self._time = None
    self.labels = []
    self.videoFPS = videoFPS

  def getLabels(self):
      return self.labels

  def processFrame(self, vehicles, lines):
    # # TODO we can the equations from the LandLineDetector.
    # if len(lines) > 10000: # for code folding in the editor

    leftXB = lines[0][0]
    leftYB = lines[0][1]
    leftXT = lines[0][2]
    leftYT = lines[0][3]

    rightXB = lines[1][0]
    rightYB = lines[1][1]
    rightXT = lines[1][2]
    rightYT = lines[1][3]

    leftSlope = (leftYT - leftYB) / (leftXT - leftXB)
    leftInt = leftYB - (leftSlope * leftXB)

    rightSlope = (rightYT - rightYB) / (rightXT - rightXB)
    rightInt = rightYB - (rightSlope * rightXB)

    # This section finds all the boxes within the current lane
    # THINGS TO DO IN THIS SECTION:
    #     detect boxes that are half in the lane on left and right
    #     detect boxes completely out of lane on left and right

    boxesOutLaneLeft = []
    boxesOnLeftLane = []
    boxesInLane = []
    boxesOnRightLane = []
    boxesOutLaneRight = []

    boxIndex = 0

    for vehicle in vehicles:
      box = vehicle.box
      lInsideLeftEdge = False
      rInsideLeftEdge = False
      lInsideRightEdge = False
      rInsideRightEdge = False

      leftX = box[0]
      rightX = box[2]
      Y = box[3]

      lEdgeLeftLaneY = leftSlope*leftX + leftInt
      lEdgeRightLaneY = rightSlope*leftX + rightInt

      rEdgeLeftLaneY = leftSlope*rightX + leftInt
      rEdgeRightLaneY = rightSlope*rightX + rightInt

      if Y > lEdgeLeftLaneY:
        lInsideLeftEdge = True

      if Y > lEdgeRightLaneY:
        lInsideRightEdge = True

      if Y > rEdgeLeftLaneY:
        rInsideLeftEdge = True

      if Y > rEdgeRightLaneY:
        rInsideRightEdge = True

      if lInsideLeftEdge and rInsideRightEdge:
        boxesInLane.append(box)

      if not lInsideLeftEdge and rInsideLeftEdge:
        boxesOnLeftLane.append(box)

      if not lInsideLeftEdge and not rInsideLeftEdge:
        boxesOutLaneLeft.append(box)

      if not rInsideRightEdge and lInsideRightEdge:
        boxesOnRightLane.append(box)

      if not rInsideRightEdge and not lInsideRightEdge:
        boxesOutLaneRight.append(box)

    # Actually produce the labels
    # THINGS TO DO IN THIS SECTION:
    #     Use ID of current Target object to find out which list its in:
    #     The first _time the target object leaves the lane, begin decreasing the new event timer
    #     If the new event timer reaches 0 and the target object has not returned to the lane, produce a new label
    #     Once the target object leaves the lane, start the new event timer
    #     When timer reaches 0 produce evtEnd label and set currentTargetObject to None

    # This section is used to determine the targetObject
    if currentTargetObject is None:
      y = 0
      targetFound = False
      for box in boxesInLane:
        # finds the closest target to the host vehicle
        if box[3] > y:
          currentTargetObjectBox = box
          targetFound = True

      if targetFound:
        newLabel = ("rightTO", self._time)
        self.labels.append(newLabel)
        lastLabelProduced = "rightTO"

    if lastLabelProduced == "rightTO":
      pass

    if lastLabelProduced == "objTurnOff":
      # Check how much time is left on new event timer
      # Also check that box is still on one of the lanes
      pass

    if lastLabelProduced == "evtEnd":
      pass
