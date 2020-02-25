from VehicleTracker import Vehicle

def isLeft(a, b, c):
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0

class LabelGenerator:
  def __init__(self, videoFPS):
    self.buffer = 1
    self.currentTargetObject = None
    self.newPotentialTarget = None
    self.newEventTimer = self.buffer
    self.newTargetTimer = self.buffer
    self.lastLabelProduced = None
    self.targetDirection = None
    self.lastTargetPos = None
    self._time = None
    self.labels = []
    self.videoFPS = videoFPS

  def getLabels(self):
    return self.labels

  def processFrame(self, vehicles, lines, frameIndex):
    if len(lines) != 2:
      return

    self._time = frameIndex / self.videoFPS

    leftSegments, _ = lines[0]
    rightSegments, _ = lines[1]

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

      x1,y1,x2,y2 = box
      leftX = min(x1,x2)
      rightX = max(x1,x2)
      bottomY = max(y1,y2)

      lInsideLeftLaneLine = False
      rInsideLeftLaneLine = False
      for (sx1, sy1, sx2, sy2) in leftSegments:
        sx2 = max(sx1, sx2)
        sx1 = min(sx1, sx2)
        sy2 = min(sy1, sy2)
        sy1 = max(sy1, sy2)
        if leftX >= sx1 and leftX <= sx2 and bottomY <= sy1 and bottomY >= sy2:
          lInsideLeftLaneLine = not isLeft((sx1,sy1),(sx2,sy2),(leftX,bottomY))
        elif leftX >= sx2 and bottomY <= sy1 and bottomY >= sy2:
          lInsideLeftLaneLine = False
        elif leftX <= sx1 and bottomY <= sy1 and bottomY >= sy2:
          lInsideLeftLaneLine = True

        if rightX >= sx1 and rightX <= sx2 and bottomY <= sy1 and bottomY >= sy2:
          rInsideLeftLaneLine = not isLeft((sx1,sy1),(sx2,sy2),(rightX,bottomY))
        elif rightX >= sx2 and bottomY <= sy1 and bottomY >= sy2:
          rInsideLeftLaneLine = False
        elif rightX <= sx1 and bottomY <= sy1 and bottomY >= sy2:
          rInsideLeftLaneLine = True

        if rInsideLeftLaneLine or lInsideLeftLaneLine:
          break

      lInsideRightLaneLine = False
      rInsideRightLaneLine = False
      for (sx1, sy1, sx2, sy2) in rightSegments:
        sx2 = max(sx1, sx2)
        sx1 = min(sx1, sx2)
        sy2 = min(sy1, sy2)
        sy1 = max(sy1, sy2)
        if leftX >= sx1 and leftX <= sx2 and bottomY <= sy1 and bottomY >= sy2:
          lInsideRightLaneLine = isLeft((sx1,sy1),(sx2,sy2),(leftX,bottomY))
        elif leftX <= sx1 and bottomY <= sy1 and bottomY >= sy2:
          lInsideRightLaneLine = True
        elif leftX >= sx2 and bottomY <= sy1 and bottomY >= sy2:
          lInsideRightLaneLine = False

        if rightX >= sx1 and rightX <= sx2 and bottomY <= sy1 and bottomY >= sy2:
          rInsideRightLaneLine = isLeft((sx1,sy1),(sx2,sy2),(rightX,bottomY))
        elif rightX <= sx1 and bottomY <= sy1 and bottomY >= sy2:
          rInsideRightLaneLine = True
        elif rightX >= sx2 and bottomY <= sy1 and bottomY >= sy2:
          rInsideRightLaneLine = False
        
        if rInsideRightLaneLine or lInsideRightLaneLine:
          break

      if lInsideLeftLaneLine and rInsideRightLaneLine:
        vehiclesInLane.append(vehicle)
      elif not lInsideLeftLaneLine and rInsideLeftLaneLine:
        vehiclesOnLeftLane.append(vehicle)
      elif not lInsideLeftLaneLine and not rInsideLeftLaneLine:
        vehiclesOutLaneLeft.append(vehicle)
      elif not rInsideRightLaneLine and lInsideRightLaneLine:
        vehiclesOnRightLane.append(vehicle)
      elif not rInsideRightLaneLine and not lInsideRightLaneLine:
        vehiclesOutLaneRight.append(vehicle)

    #leftXB = lines[0][0][0]
    #leftYB = lines[0][0][1]
    #leftXT = lines[0][0][2]
    #leftYT = lines[0][0][3]

    #rightXB = lines[1][0][0]
    #rightYB = lines[1][0][1]
    #rightXT = lines[1][0][2]
    #rightYT = lines[1][0][3]

    #leftSlope = (leftYT - leftYB) / (leftXT - leftXB)
    #leftInt = leftYB - (leftSlope * leftXB)

    #rightSlope = (rightYT - rightYB) / (rightXT - rightXB)
    #rightInt = rightYB - (rightSlope * rightXB)

    ## This section finds all the boxes within the current lane
    ## THINGS TO DO IN THIS SECTION:
    ##     detect boxes that are half in the lane on left and right
    ##     detect boxes completely out of lane on left and right

    #vehiclesOutLaneLeft = []
    #vehiclesOnLeftLane = []
    #vehiclesInLane = []
    #vehiclesOnRightLane = []
    #vehiclesOutLaneRight = []

    #boxIndex = 0

    #for vehicle in vehicles:
    #  box = vehicle.box
    #  lInsideLeftEdge = False
    #  rInsideLeftEdge = False
    #  lInsideRightEdge = False
    #  rInsideRightEdge = False

    #  leftX = box[0]
    #  rightX = box[2]
    #  Y = box[3]

    #  lEdgeLeftLaneY = leftSlope*leftX + leftInt
    #  lEdgeRightLaneY = rightSlope*leftX + rightInt

    #  rEdgeLeftLaneY = leftSlope*rightX + leftInt
    #  rEdgeRightLaneY = rightSlope*rightX + rightInt

    #  if Y > lEdgeLeftLaneY:
    #    lInsideLeftEdge = True

    #  if Y > lEdgeRightLaneY:
    #    lInsideRightEdge = True

    #  if Y > rEdgeLeftLaneY:
    #    rInsideLeftEdge = True

    #  if Y > rEdgeRightLaneY:
    #    rInsideRightEdge = True

    #  if lInsideLeftEdge and rInsideRightEdge:
    #    vehiclesInLane.append(vehicle)

    #  if not lInsideLeftEdge and rInsideLeftEdge:
    #    vehiclesOnLeftLane.append(vehicle)

    #  if not lInsideLeftEdge and not rInsideLeftEdge:
    #    vehiclesOutLaneLeft.append(vehicle)

    #  if not rInsideRightEdge and lInsideRightEdge:
    #    vehiclesOnRightLane.append(vehicle)

    #  if not rInsideRightEdge and not lInsideRightEdge:
    #    vehiclesOutLaneRight.append(vehicle)




    '''
    Finds the closest vehicle in our lane
    '''
    y = 0
    closestTarget = None
    for vehicle in vehiclesInLane:
      # finds the closest target to the host vehicle
      if vehicle.box[3] > y:
        y = vehicle.box[3]
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
    elif closestTarget is not None:
      if self.currentTargetObject.id == closestTarget.id:
        targetInLane = True

    '''
    This section handles when a target object is in the host vehicle lane
    '''
    if self.lastLabelProduced == "rightTO":

      # Current target object is closest in lane
      if targetInLane:
        self.newEventTimer = self.buffer
        self.lastTargetPos = "In Lane"

      else:
        # if this is the first time it has left the host lane
        if self.newEventTimer == self.buffer:
          self.newEventTimer = self.buffer - 1
          for vehicle in vehiclesOnLeftLane:
            if vehicle.id == self.currentTargetObject.id:
              self.targetDirection = "Left"


          for vehicle in vehiclesOnRightLane:
            if vehicle.id == self.currentTargetObject.id:
              self.targetDirection = "Right"


        # if the target object has already started leaving the lane
        elif self.newEventTimer > 0 and self.newEventTimer < self.buffer:
          if self.targetDirection == "Left":
            for vehicle in vehiclesOnLeftLane:
              if vehicle.id == self.currentTargetObject.id:
                self.newEventTimer -= 1


          if self.targetDirection == "Right":
            for vehicle in vehiclesOnRightLane:
              if vehicle.id == self.currentTargetObject.id:
                self.newEventTimer -= 1


        # if the target has left the lane
        else:
          newLabel = ("cutout", self._time)
          self.labels.append(newLabel)
          self.lastLabelProduced = "cutout"
          self.newEventTimer = self.buffer
          if self.targetDirection == "Left":
            self.lastTargetPos = "On Left Line"
          if self.targetDirection == "Right":
            self.lastTargetPos = "On Right Line"



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
        self.newEventTimer = self.buffer


      # Object is in neighbor lane
      elif inOutsideLane:
        if self.newEventTimer > 0:
          self.newEventTimer -= 1


        else:
          newLabel = ("evtEnd", self._time)
          self.labels.append(newLabel)
          self.lastLabelProduced = "evtEnd"
          self.newEventTimer = self.buffer
          self.currentTargetObject = None
          self.lastTargetPos = None
          self.targetDirection = None


      else:
        if self.newEventTimer > 0:
          self.newEventTimer -= 1


        else:
          self.newEventTimer = self.buffer
          del self.labels[-1]
          self.lastLabelProduced = "rightTO"
          self.lastTargetPos = "In Lane"
          self.targetDirection = None


    '''
    This section will track and handle new cars cutting into our lane
    '''
    closerSideTarget = None
    if targetInLane:
      for vehicle in vehiclesOnLeftLane:
        # finds the closest target to the host vehicle
        if vehicle.box[3] > y:
          closerSideTarget = vehicle
          y = vehicle.box[3]

      for vehicle in vehiclesOnRightLane:
        if vehicle.box[3] > y:
          closerSideTarget = vehicle
          y = vehicle.box[3]


    if self.newPotentialTarget is None:
      if closerSideTarget is not None:
        self.newPotentialTarget = closerSideTarget
        self.newTargetTimer = self.buffer-1

    elif self.newPotentialTarget is not None and self.lastLabelProduced != "cutin":
      if closerSideTarget is None:
        self.newPotentialTarget = None
        self.newTargetTimer = self.buffer
      else:
        if closerSideTarget.id == self.newPotentialTarget.id:
          if self.newTargetTimer > 0:
            self.newTargetTimer -= 1
          else:
            newLabel = ("cutin", self._time)
            self.labels.append(newLabel)
            self.lastLabelProduced = "cutin"
            self.newTargetTimer = self.buffer

        else:
          self.newPotentialTarget = closerSideTarget
          self.newTargetTimer = self.buffer - 1

    else:
      if closerSideTarget is None:
        newTargetInLane = False
        for vehicle in vehiclesInLane:
          if vehicle.id == self.newPotentialTarget.id:
            newTargetInLane = True

        if newTargetInLane:
          if self.newTargetTimer > 0:
            self.newTargetTimer -= 1

          else:
            newLabel = ("evtEnd", self._time)
            self.labels.append(newLabel)
            self.currentTargetObject = self.newPotentialTarget
            self.newPotentialTarget = None
            newLabel2 = ("rightTO", self._time)
            self.labels.append(newLabel2)
            self.lastLabelProduced = "rightTO"
            self.lastTargetPos = "In Lane"

        else:
          if self.newTargetTimer > 0:
            self.newTargetTimer -= 1

          else:
            del self.labels[-1]
            self.newPotentialTarget = None
            self.newTargetTimer = self.buffer

      else:
        if closerSideTarget.id == self.newPotentialTarget.id:
          self.newTargetTimer = self.buffer

        else:
          self.newPotentialTarget = closerSideTarget
          self.newTargetTimer = self.buffer



