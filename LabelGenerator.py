from VehicleTracker import Vehicle

class LabelGenerator:
  def __init__(self, videoFPS):
    self.currentTargetObject = None
    self.newPotentialTarget = None
    self.newEventTimer = 10
    self.lastLabelProduced = None
    self.targetDirection = None
    self.lastTargetPos = None
    self._time = None
    self.labels = []
    self.videoFPS = videoFPS

  def getLabels(self):
      return self.labels

  def processFrame(self, vehicles, lines):

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

    vehiclesOutLaneLeft = []
    vehiclesOnLeftLane = []
    vehiclesInLane = []
    vehiclesOnRightLane = []
    vehiclesOutLaneRight = []

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
        vehiclesInLane.append(vehicle)

      if not lInsideLeftEdge and rInsideLeftEdge:
        vehiclesOnLeftLane.append(vehicle)

      if not lInsideLeftEdge and not rInsideLeftEdge:
        vehiclesOutLaneLeft.append(vehicle)

      if not rInsideRightEdge and lInsideRightEdge:
        vehiclesOnRightLane.append(vehicle)

      if not rInsideRightEdge and not lInsideRightEdge:
        vehiclesOutLaneRight.append(vehicle)


    '''
    Finds the closest vehicle in our lane
    '''
    y = 0
    closestTarget = None
    for vehicle in vehiclesInLane:
      # finds the closest target to the host vehicle
      if vehicle.box[3] > y:
        closestTarget = vehicle

    '''
    Sets a target object if there is no current target object
    Otherwise checks that the current target object is still the 
    closest vehicle in lane
    '''
    targetInLane = False
    if self.currentTargetObject is None:
      if closestTarget is not None:
        self.currentTargetObject = closestTarget
        newLabel = ("rightTO", self._time)
        self.labels.append(newLabel)
        self.lastLabelProduced = "rightTO"
        self.lastTargetPos = "In Lane"
        targetInLane = True
    else:
      if self.currentTargetObject.id == closestTarget.id:
        targetInLane = True

    '''
    This section handles when a target object is in the host vehicle lane
    '''
    if self.lastLabelProduced == "rightTO":

      # Current target object is closest in lane
      if targetInLane:
        self.newEventTimer = 10
        self.lastTargetPos = "In Lane"
        return

      # if this is the first time it has left the host lane
      if self.newEventTimer == 10:
        self.newEventTimer = 9
        for vehicle in vehiclesOnLeftLane:
          if vehicle.id == self.currentTargetObject.id:
            self.targetDirection = "Left"
            return

        for vehicle in vehiclesOnRightLane:
          if vehicle.id == self.currentTargetObject.id:
            self.targetDirection = "Right"
            return

      # if the target object has already started leaving the lane
      elif self.newEventTimer > 0 and self.newEventTimer < 10:
        if self.targetDirection == "Left":
          for vehicle in vehiclesOnLeftLane:
            if vehicle.id == self.currentTargetObject.id:
              self.newEventTimer -= 1
              return

        if self.targetDirection == "Right":
          for vehicle in vehiclesOnRightLane:
            if vehicle.id == self.currentTargetObject.id:
              self.newEventTimer -= 1
              return

      # if the target has left the lane
      else:
        newLabel = ("cutout", self._time)
        self.labels.append(newLabel)
        self.lastLabelProduced = "cutout"
        self.newEventTimer = 10
        if self.targetDirection == "Left":
          self.lastTargetPos = "On Left Line"
        if self.targetDirection == "Right":
          self.lastTargetPos = "On Right Line"
        return


    '''
    This Section handles when the current target object has cut out of 
    the host vehicles lane
    '''
    if self.lastLabelProduced == "cutout":
      stillOnEdge = False
      inOutsideLane = False
      if self.lastTargetPos == "On Left Line":
        for vehicle in vehiclesOnLeftLane:
          if vehicle.id == self.currentTargetObject.id:
            stillOnEdge = True
        for vehicle in vehiclesOutLaneLeft:
          if vehicle.id == self.currentTargetObject.id:
            inOutsideLane = True

      if self.lastTargetPos == "On Right Line":
        for vehicle in vehiclesOnRightLane:
          if vehicle.id == self.currentTargetObject.id:
            stillOnEdge = True
        for vehicle in vehiclesOutLaneRight:
          if vehicle.id == self.currentTargetObject.id:
            inOutsideLane = True

      # Object is still on the lane line
      if stillOnEdge:
        self.newEventTimer = 10
        return

      # Object is in neighbor lane
      if inOutsideLane:
        if self.newEventTimer > 0:
          self.newEventTimer -= 1
          return

        else:
          newLabel = ("evtEnd", self._time)
          self.labels.append(newLabel)
          self.lastLabelProduced = "evtEnd"
          self.newEventTimer = 10
          self.currentTargetObject = None
          self.lastTargetPos = None
          self.targetDirection = None
          return

      if targetInLane:
        if self.newEventTimer > 0:
          self.newEventTimer -= 1
          return

        else:
          self.newEventTimer = 10
          del self.labels[-1]
          self.lastLabelProduced = "rightTO"
          self.lastTargetPos = "In Lane"
          self.targetDirection = None

