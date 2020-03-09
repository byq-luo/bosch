from VehicleTracker import Vehicle

def isLeft(a, b, c):
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0

class LabelGenerator:
  def __init__(self, videoFPS):
    self.buffer = 10
    self.cancelBuffer = 3
    self.targetLostBuffer = 5
    self.targetLostTimer = self.targetLostBuffer
    self.cancelTimer = self.cancelBuffer
    self.cancelCutinTimer = self.cancelBuffer
    self.currentTargetObject = None
    self.newPotentialTarget = None
    self.newCutinTarget = None
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
    targetFound = False
    for vehicle in vehicles:
      if self.currentTargetObject is not None:
        if vehicle.id == self.currentTargetObject.id:
          targetFound = True

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
    Ensure the target object still exists, resets if target object disappears
    '''
    if self.currentTargetObject is not None:
      if targetFound:
        self.targetLostTimer = self.targetLostBuffer
      else:
        if self.targetLostTimer > 0:
          self.targetLostTimer -= 1
        else:
          self.currentTargetObject = None
          self.newEventTimer = self.buffer
          self.cancelTimer = self.cancelBuffer
          self.targetDirection = None
          self.lastTargetPos = None
          self.targetLostTimer = self.targetLostBuffer

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

        # Begin countdown making the closestTarget the new currentTarget
        if self.newEventTimer == self.buffer:
          self.newPotentialTarget = closestTarget
          self.newEventTimer -= 1

        # Handles when a vehicle is becoming the new currentTarget
        elif self.newEventTimer > 0:
          if self.newPotentialTarget.id == closestTarget.id:
            self.newEventTimer -= 1
          else:
            if self.cancelTimer > 0:
              self.cancelTimer -= 1
            else:
              self.newPotentialTarget = None
              self.newEventTimer = self.buffer
              self.cancelTimer = self.cancelBuffer

        # Handles when a vehicle has become the new currentTarget
        else:
          self.currentTargetObject = closestTarget
          newLabel = ("rightTO", self._time)
          self.labels.append(newLabel)
          self.lastLabelProduced = "rightTO"
          self.lastTargetPos = "In Lane"
          targetInLane = True
          self.newPotentialTarget = None

    # Checks to see if currentTarget object is still in the lane
    elif closestTarget is not None:
      if self.currentTargetObject.id == closestTarget.id:
        targetInLane = True

    '''
    This section handles when a target object is in the host vehicle lane
    '''
    if self.lastLabelProduced == "rightTO":

      # Current target object is closest in lane
      if targetInLane:
        if self.newEventTimer < self.buffer:
          if self.cancelTimer > 0:
            self.cancelTimer -= 1
          else:
            self.newEventTimer = self.buffer
            self.cancelTimer = self.cancelBuffer
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
          self.cancelTimer = self.cancelBuffer
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
        if self.newEventTimer < self.buffer:
          if self.cancelTimer > 0:
            self.cancelTimer -= 1
          else:
            self.newEventTimer = self.buffer
            self.cancelTimer = self.cancelBuffer


      # Target object is in neighbor lane
      elif inOutsideLane:
        # Still in the process of leaving lane
        if self.newEventTimer > 0:
          self.newEventTimer -= 1

        # Has left the lane
        else:
          newLabel = ("evtEnd", self._time)
          self.labels.append(newLabel)
          self.lastLabelProduced = "evtEnd"
          self.newEventTimer = self.buffer
          self.cancelTimer = self.cancelBuffer
          self.currentTargetObject = None
          self.lastTargetPos = None
          self.targetDirection = None

      # Object in host lane
      else:
        if self.newEventTimer > 0:
          self.newEventTimer -= 1

        else:
          self.newEventTimer = self.buffer
          self.cancelTimer = self.cancelBuffer
          del self.labels[-1]
          self.lastLabelProduced = "rightTO"
          self.lastTargetPos = "In Lane"
          self.targetDirection = None


    '''
    This section will track and handle new cars cutting into our lane
    '''
    closerSideTarget = None
    y = 0

    for vehicle in vehiclesInLane:
      if vehicle.box[3] > y:
        closerSideTarget = vehicle
        y = vehicle.box[3]

    for vehicle in vehiclesOnLeftLane:
      # finds the closest target to the host vehicle
      if vehicle.box[3] > y:
        closerSideTarget = vehicle
        y = vehicle.box[3]

    for vehicle in vehiclesOnRightLane:
      if vehicle.box[3] > y:
        closerSideTarget = vehicle
        y = vehicle.box[3]

    # Identify a new target cutting in to host lane
    if self.newCutinTarget is None:
      if closerSideTarget is not None and self.currentTargetObject is not None:
        if closerSideTarget.id != self.currentTargetObject.id:
          self.newCutinTarget = closerSideTarget
          self.newTargetTimer = self.buffer-1

    # Handles when a new target is in the process of cutting in to host lane
    elif self.newCutinTarget is not None and self.lastLabelProduced != "cutin":
      if closerSideTarget is None:
        if self.cancelCutinTimer > 0:
          self.cancelCutinTimer -= 1
        else:
          self.newCutinTarget = None
          self.cancelCutinTimer = self.cancelBuffer
          self.newTargetTimer = self.buffer

      else:
        if closerSideTarget.id == self.newCutinTarget.id:
          if self.newTargetTimer > 0:
            self.newTargetTimer -= 1
          else:
            newLabel = ("cutin", self._time)
            self.labels.append(newLabel)
            self.lastLabelProduced = "cutin"
            self.newTargetTimer = self.buffer

        else:
          if self.cancelCutinTimer > 0:
            self.cancelCutinTimer -= 1
          else:
            self.newCutinTarget = None
            self.cancelCutinTimer = self.cancelBuffer
            self.newTargetTimer = self.buffer

    # Handles when a target has cut into the lane
    else:
      if closerSideTarget is not None:
        newTargetInLane = False
        for vehicle in vehiclesInLane:
          if vehicle.id == self.newCutinTarget.id:
            newTargetInLane = True

        if newTargetInLane:
          if self.newTargetTimer > 0:
            self.newTargetTimer -= 1

          else:
            newLabel = ("evtEnd", self._time)
            self.labels.append(newLabel)
            self.currentTargetObject = self.newCutinTarget
            self.newCutinTarget = None
            newLabel2 = ("rightTO", self._time)
            self.labels.append(newLabel2)
            self.lastLabelProduced = "rightTO"
            self.lastTargetPos = "In Lane"
            self.newTargetTimer = self.buffer
            self.cancelCutinTimer = self.cancelBuffer

        else:
          if closerSideTarget.id == self.newCutinTarget.id:
            if self.newTargetTimer < self.buffer:
              if self.cancelCutinTimer > 0:
                self.cancelCutinTimer -= 1
              else:
                self.newTargetTimer = self.buffer
                self.cancelCutinTimer = self.cancelBuffer

          # When the car cutting in is no longer the closer side target
          else:
            pass

      else:
        if self.newTargetTimer > 0:
          self.newTargetTimer -= 1

        else:
          del self.labels[-1]
          self.newCutinTarget = None
          self.newTargetTimer = self.buffer
          self.cancelCutinTimer = self.cancelBuffer




