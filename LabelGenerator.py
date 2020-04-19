from VehicleTracker import Vehicle

def isLeft(a, b, c):
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0

class LabelGenerator:
  def __init__(self, videoFPS):
    self.buffer = 30
    self.cancelBuffer = 15
    self.targetLostBuffer = 15
    self.targetLostTimer = self.targetLostBuffer
    self.cancelTimer = self.cancelBuffer
    self.cancelCutinTimer = self.cancelBuffer
    self.currentTargetObject = None
    self.newPotentialTarget = None
    self.newCutinTarget = None
    self.cutoutTimer = self.buffer
    self.cancelCutoutTimer = self.cancelBuffer
    self.reverseCutoutTimer = self.buffer
    self.cancelReverseCutout = self.cancelBuffer
    self.newEventTimer = self.buffer
    self.newTargetTimer = self.buffer
    self.laneChangeTimer = self.buffer
    self.cancelLaneChangeTimer = self.buffer
    self.lastLabelProduced = None
    self.targetDirection = None
    self.lastTargetPos = None
    self._time = None
    self.label_time = None
    self.cutinLabelTime = None
    self.labels = []
    self.videoFPS = videoFPS
    self.seconds = 0
    self.currentLeftX = 0
    self.currentRightX = 0
    self.prevLeftX = None
    self.prevRightX = None
    self.laneChangeBuffer = 100
    self.laneChangeDir = None
    self.endBuffer = 130
    self.endTimer = self.endBuffer
    self.lastTO = None
    self.laneChangeEndBuffer = 45
    self.laneChangeEndTimer = self.laneChangeEndBuffer
    self.lastTargetX = None
    self.cancelEndBuffer = 10
    self.cancelEndTimer = self.cancelEndBuffer


  def getLabels(self):
    return self.labels


  def processFrame(self, vehicles, lines, frameIndex):
    if len(lines) != 2:
      return

    self._time = frameIndex / self.videoFPS

    if frameIndex % 30 == 0:
      self.seconds += 1


    leftSegments = lines[0]
    rightSegments = lines[1]

    missingLane = False
    if len(leftSegments) == 0 or len(rightSegments) == 0:
      missingLane = True

    if self.lastLabelProduced == "end":
      if missingLane:
        if self.cancelEndTimer > 0:
          self.cancelEndTimer -= 1
        else:
          self.endTimer = self.endBuffer
          self.cancelEndTimer = self.cancelEndBuffer
      else:
        if self.endTimer > 0:
          self.endTimer -= 1
        else:
          eventTime = self._time - (self.endBuffer / self.videoFPS)
          newLabel = ("evtEnd", eventTime)
          self.labels.append(newLabel)
          self.lastLabelProduced = "evtEnd"
          self.newEventTimer = self.buffer
          self.cancelTimer = self.cancelBuffer
          self.currentTargetObject = None
          self.lastTargetPos = None
          self.targetDirection = None
          self.cancelEndTimer = self.cancelEndBuffer
      return
    else:
      if missingLane:
        if self.endTimer > 0:
          self.endTimer -= 1
        else:
          #eventTime = self._time - (self.endBuffer / self.videoFPS)
          newLabel = ("end", self._time)
          self.labels.append(newLabel)
          self.lastLabelProduced = "end"
          return
      else:
        if self.cancelEndTimer > 0:
          self.cancelEndTimer -= 1
        else:
          self.endTimer = self.endBuffer
          self.cancelEndTimer = self.cancelEndBuffer


    vehiclesOutLaneLeft = []
    vehiclesOnLeftLane = []
    vehiclesInLane = []
    vehiclesOnRightLane = []
    vehiclesOutLaneRight = []

    boxIndex = 0
    targetFound = False

    firstLeftX = None
    firstRightX = None
    for vehicle in vehicles:

      box = vehicle.box

      x1,y1,x2,y2 = box
      leftX = min(x1,x2)
      rightX = max(x1,x2)

      midX = (rightX + leftX) / 2
      bottomY = max(y1,y2)

      if self.currentTargetObject is not None:
        if vehicle.id == self.currentTargetObject.id:
          targetFound = True
          self.lastTargetX = midX

      leftSeg = None
      for (sx1, sy1, sx2, sy2) in leftSegments:
        if bottomY <= sy1 and bottomY >= sy2:
          leftSeg = sx1
        if 300 <= sy1 and 300 >= sy2:
          firstLeftX = sx1

      rightSeg = None
      for (sx1, sy1, sx2, sy2) in rightSegments:
        if bottomY <= sy1 and bottomY >= sy2:
          rightSeg = sx1
        if 300 <= sy1 and 300 >= sy2:
          firstRightX = sx1

      if leftSeg is not None and rightSeg is not None:
        lLineX = leftSeg
        rLineX = rightSeg
        laneWidth = rLineX - lLineX
        laneBuffer = laneWidth / 4
        if midX < (lLineX - (laneBuffer/2)):
          vehiclesOutLaneLeft.append(vehicle)
        elif midX > (lLineX - (laneBuffer/2)) and midX < (lLineX + laneBuffer):
          vehiclesOnLeftLane.append(vehicle)
        elif midX > (lLineX + laneBuffer) and midX < (rLineX - laneBuffer):
          vehiclesInLane.append(vehicle)
        elif midX > (rLineX - laneBuffer) and midX < (rLineX + (laneBuffer/2)):
          vehiclesOnRightLane.append(vehicle)
        elif midX > (rLineX + (laneBuffer/2)):
          vehiclesOutLaneRight.append(vehicle)
        else:
          pass


    if frameIndex % 30 == 0:
      print(self.seconds)
      laneCounts = [len(vehiclesOutLaneLeft), len(vehiclesOnLeftLane), len(vehiclesInLane),
                    len(vehiclesOnRightLane), len(vehiclesOutLaneRight)]
      print(laneCounts)
      print(self.currentTargetObject)
      self.seconds += 1
      print(self.lastTargetX)


    if self.seconds > 34 and self.seconds < 40:
      frame = frameIndex % self.videoFPS
      print("Second: ", self.seconds, " Frame: ", frame)
      print("Current Target object: ", self.currentTargetObject)
      print("Current Cutin Target: ", self.newCutinTarget)
      print("Outside Left")
      for v in vehiclesOutLaneLeft:
        print(v.id)
      print("On Left")
      for v in vehiclesOnLeftLane:
        print(v.id)
      print("In Lane")
      for v in vehiclesInLane:
        print(v.id)


    '''
    Checks if the host is changing lanes
    '''
    leftChange = False
    rightChange = False
    if firstLeftX is None or firstRightX is None:
      firstLeftX = self.currentLeftX
      firstRightX = self.currentRightX

    leftDx = firstLeftX - self.currentLeftX
    rightDx = firstRightX - self.currentRightX
    if abs(leftDx) > self.laneChangeBuffer:
      leftChange = True
    if abs(rightDx) > self.laneChangeBuffer:
      rightChange = True


    if self.lastLabelProduced != "lcRel":
      self.currentRightX = firstRightX
      self.currentLeftX = firstLeftX

      if self.laneChangeDir is None and self.laneChangeTimer == self.buffer:

        if leftChange is True and rightChange is False:
          self.laneChangeDir = "Left"
          self.prevLeftX = self.currentLeftX
          self.prevRightX = self.currentRightX
          self.laneChangeTimer -= 1
        elif rightChange is True and leftChange is False:
          self.laneChangeDir = "Right"
          self.prevRightX = self.currentRightX
          self.prevLeftX = self.currentLeftX
          self.laneChangeTimer -= 1

      elif self.laneChangeDir == "Left":
        if leftChange is False:
          if self.laneChangeTimer > 0:
            self.laneChangeTimer -= 1
          else:
            eventTime = self._time - (self.buffer / self.videoFPS)
            self.labels.append(("lcRel", eventTime))
            print("The target this lcRel is acting on is ", self.currentTargetObject)
            self.lastLabelProduced = "lcRel"
            self.laneChangeTimer = self.buffer
            self.cancelLaneChangeTimer = self.cancelBuffer
            self.newCutinTarget = None
            self.newTargetTimer = self.buffer
            self.cancelCutinTimer = self.cancelBuffer

      elif self.laneChangeDir == "Right":
        if rightChange is False:
          if self.laneChangeTimer > 0:
            self.laneChangeTimer -= 1
          else:
            eventTime = self._time - (self.buffer / self.videoFPS)
            self.labels.append(("lcRel", eventTime))
            print("The target this lcRel is acting on is ", self.currentTargetObject)
            self.lastLabelProduced = "lcRel"
            self.laneChangeTimer = self.buffer
            self.cancelLaneChangeTimer = self.cancelBuffer
            self.laneChangeDir = None
            self.newCutinTarget = None
            self.newTargetTimer = self.buffer
            self.cancelCutinTimer = self.cancelBuffer

    else:
      if self.laneChangeEndTimer > 0:
        self.laneChangeEndTimer -= 1
      else:
        print("In evtEnd for lcRel")
        eventTime = self._time - (self.buffer / self.videoFPS)
        newLabel = ("evtEnd", eventTime)
        print("The target this lcRel evtEnd is acting on is ", self.currentTargetObject)
        self.labels.append(newLabel)
        self.currentTargetObject = None
        self.prevLeftX = None
        self.prevRightX = None
        self.laneChangeDir = None
        self.cancelLaneChangeTimer = self.cancelBuffer
        self.laneChangeTimer = self.buffer
        self.lastLabelProduced = "evtEnd"
        self.laneChangeEndTimer = self.laneChangeEndBuffer


    '''
    Ensure the target object still exists, resets if target object disappears
    '''
    if self.currentTargetObject is not None and self.lastLabelProduced != "cutin":
      if targetFound:
        self.targetLostTimer = self.targetLostBuffer
      else:
        if self.targetLostTimer > 0:
          self.targetLostTimer -= 1
        else:
          self.lastTO = self.currentTargetObject
          self.currentTargetObject = None
          self.newEventTimer = self.buffer
          self.cancelTimer = self.cancelBuffer
          self.targetDirection = None
          self.lastTargetPos = None
          self.targetLostTimer = self.targetLostBuffer
          self.newTargetTimer = self.buffer
          self.cancelCutinTimer = self.cancelBuffer
          self.newCutinTarget = None
          eventTime = self._time - (self.buffer / self.videoFPS)
          if self.lastTargetX < 100 or self.lastTargetX > 500:
            self.labels.append(("objTurnOff", eventTime))
            print("The target this objTurnOff is acting on is ", self.currentTargetObject)
            self.labels.append(("evtEnd", self._time))
            self.lastLabelProduced = "evtEnd"
            self.lastTO = None




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
    potentialInLane = False
    targetInLane = False
    if self.currentTargetObject is None and self.lastLabelProduced != "cutin":
      if closestTarget is not None:
        # Begin countdown making the closestTarget the new currentTarget
        if self.newPotentialTarget is None:
          self.label_time = self._time
          self.newPotentialTarget = closestTarget
          self.newEventTimer -= 1

        # Handles when a vehicle is becoming the new currentTarget
        elif self.newEventTimer > 0 and self.newEventTimer < self.buffer:
          if self.newPotentialTarget.id == closestTarget.id:
            potentialInLane = True
            self.newEventTimer -= 1
          else:
            if self.cancelTimer > 0:
              self.cancelTimer -= 1
            else:
              self.newPotentialTarget = None
              self.newEventTimer = self.buffer
              self.cancelTimer = self.cancelBuffer
              self.label_time = None

        # Handles when a vehicle has become the new currentTarget
        else:
          if self.lastTO is not None:
            self.lastTO = closestTarget
          else:
            eventTime = self._time - (self.buffer / self.videoFPS)
            newLabel = ("rightTO", eventTime)
            self.labels.append(newLabel)

          self.currentTargetObject = closestTarget
          self.lastLabelProduced = "rightTO"
          print("The target this rightTO is acting on is ", self.currentTargetObject)
          self.lastTargetPos = "In Lane"
          targetInLane = True
          self.newPotentialTarget = None
          self.newEventTimer = self.buffer

    # Checks to see if currentTarget object is still in the lane
    elif closestTarget is not None and self.lastLabelProduced != "cutin":
      if self.currentTargetObject.id == closestTarget.id:
        targetInLane = True

    '''
    This section handles when a target object is in the host vehicle lane
    '''
    if self.lastLabelProduced == "rightTO" and self.currentTargetObject is not None:
      # Current target object is closest in lane
      if targetInLane:
        if self.cutoutTimer < self.buffer:
          if self.cancelCutoutTimer > 0:
            self.cancelCutoutTimer -= 1
          else:
            self.cutoutTimer = self.buffer
            self.cancelCutoutTimer = self.cancelBuffer
            self.lastTargetPos = "In Lane"


      else:
        # if this is the first time it has left the host lane
        if self.cutoutTimer == self.buffer:
          self.cutoutTimer = self.buffer - 1
          for vehicle in vehiclesOnLeftLane:
            if vehicle.id == self.currentTargetObject.id:
              self.targetDirection = "Left"

          for vehicle in vehiclesOutLaneLeft:
            print("THE VEHICLE IS ", vehicle)
            print("THE CURRENT TARGET IS ", vehicle)
            if vehicle.id == self.currentTargetObject.id:
              self.targetDirection = "Left"

          for vehicle in vehiclesOnRightLane:
            if vehicle.id == self.currentTargetObject.id:
              self.targetDirection = "Right"

          for vehicle in vehiclesOutLaneRight:
            if vehicle.id == self.currentTargetObject.id:
              self.targetDirection = "Right"


        # if the target object has already started leaving the lane
        elif self.cutoutTimer > 0 and self.cutoutTimer < self.buffer:

          if self.targetDirection == "Left":
            for vehicle in vehiclesOnLeftLane:
              if vehicle.id == self.currentTargetObject.id:
                self.cutoutTimer -= 1
            for vehicle in vehiclesOutLaneLeft:
              if vehicle.id == self.currentTargetObject.id:
                self.cutoutTimer -= 1
          if self.targetDirection == "Right":
            for vehicle in vehiclesOnRightLane:
              if vehicle.id == self.currentTargetObject.id:
                self.cutoutTimer -= 1
            for vehicle in vehiclesOutLaneRight:
              if vehicle.id == self.currentTargetObject.id:
                self.cutoutTimer -= 1


        # if the target has left the lane
        else:
          eventTime = self._time - (self.buffer / self.videoFPS)
          newLabel = ("cutout", eventTime)
          print("The target this cutout is acting on is ", self.currentTargetObject)
          self.labels.append(newLabel)
          self.lastLabelProduced = "cutout"
          self.cutoutTimer = self.buffer
          self.cancelCutoutTimer = self.cancelBuffer
          if self.targetDirection == "Left":
            self.lastTargetPos = "On Left Line"
          if self.targetDirection == "Right":
            self.lastTargetPos = "On Right Line"


    elif self.lastLabelProduced == "cutout":
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
        if self.cutoutTimer < self.buffer:
          if self.cancelCutoutTimer > 0:
            self.cancelCutoutTimer -= 1
          else:
            self.cutoutTimer = self.buffer
            self.cancelCutoutTimer = self.cancelBuffer

        if self.reverseCutoutTimer < self.buffer:
          if self.cancelReverseCutout > 0:
            self.cancelReverseCutout -= 1
          else:
            self.reverseCutoutTimer = self.buffer
            self.cancelReverseCutout = self.cancelBuffer

      # Target object is in neighbor lane
      elif inOutsideLane:
        if self.reverseCutoutTimer < self.buffer:
          if self.cancelReverseCutout > 0:
            self.cancelReverseCutout -= 1
          else:
            self.reverseCutoutTimer = self.buffer
            self.cancelReverseCutout = self.cancelBuffer

        # Still in the process of leaving lane
        if self.cutoutTimer > 0:
          self.cutoutTimer -= 1

        # Has left the lane
        else:
          eventTime = self._time - (self.buffer / self.videoFPS)
          newLabel = ("evtEnd", eventTime)
          print("The target this cutout evtEnd is acting on is ", self.currentTargetObject)
          self.labels.append(newLabel)
          self.lastLabelProduced = "evtEnd"
          self.cutoutTimer = self.buffer
          self.cancelCutoutTimer = self.cancelBuffer
          self.currentTargetObject = None
          self.lastTargetPos = None
          self.targetDirection = None
          self.lastTO = None

      # Object in host lane
      else:
        if self.reverseCutoutTimer > 0:
          self.reverseCutoutTimer -= 1

        else:
          self.cutoutTimer = self.buffer
          self.reverseCutoutTimer = self.buffer
          self.cancelCutoutTimer = self.cancelBuffer
          self.cancelReverseCutout = self.cancelBuffer
          print("DELETE CUTOUT LABEL")
          del self.labels[-1]
          self.lastLabelProduced = "rightTO"
          self.lastTargetPos = "In Lane"
          self.targetDirection = None

    '''
    This section will track and handle new cars cutting into our lane
    '''
    if self.lastLabelProduced == "lcRel" or self.lastLabelProduced == "cutout":
      return

    closerSideTarget = None
    targetInAdjacentLane = False
    y = 0

#    for vehicle in vehiclesInLane:
#      if vehicle.box[3] > y:
#        closerSideTarget = vehicle
#        y = vehicle.box[3]

    for vehicle in vehiclesOnLeftLane:
      # finds the closest target to the host vehicle
      if vehicle.box[3] > y:
        closerSideTarget = vehicle
        y = vehicle.box[3]
      if self.currentTargetObject is not None:
        if vehicle.id == self.currentTargetObject.id:
          targetInAdjacentLane = True

    for vehicle in vehiclesOnRightLane:
      if vehicle.box[3] > y:
        closerSideTarget = vehicle
        y = vehicle.box[3]
      if self.currentTargetObject is not None:
        if vehicle.id == self.currentTargetObject.id:
          targetInAdjacentLane = True

    # Identify a new target cutting in to host lane
    if targetInAdjacentLane:
      return

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
            eventTime = self._time - (self.buffer / self.videoFPS)
            newLabel = ("cutin", eventTime)
            print("The target this cutin is acting on is ", self.newCutinTarget)
            self.labels.append(newLabel)
            self.lastLabelProduced = "cutin"
            self.newTargetTimer = self.buffer
            self.cancelCutinTimer = self.cancelBuffer

        else:
          if self.cancelCutinTimer > 0:
            self.cancelCutinTimer -= 1
          else:
            self.newCutinTarget = None
            self.cancelCutinTimer = self.cancelBuffer
            self.newTargetTimer = self.buffer
            self.cutinLabelTime = None

    # Handles when a target has cut into the lane
    elif self.lastLabelProduced == "cutin":

      for vehicle in vehiclesInLane:
        if vehicle.box[3] > y:
          closerSideTarget = vehicle
          y = vehicle.box[3]

      if closerSideTarget is not None:
        newTargetInLane = False
        for vehicle in vehiclesInLane:
          if vehicle.id == self.newCutinTarget.id:
            newTargetInLane = True

        if newTargetInLane:
          if self.newTargetTimer > 0:
            self.newTargetTimer -= 1

          else:
            eventTime = self._time - (self.buffer / self.videoFPS)
            newLabel = ("evtEnd", eventTime)
            self.labels.append(newLabel)
            newLabel2 = ("rightTO", self._time)
            self.labels.append(newLabel2)
            self.currentTargetObject = self.newCutinTarget
            print("The target this cutin evtEnd is acting on is ", self.newCutinTarget)
            self.newCutinTarget = None
            self.lastLabelProduced = "rightTO"
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
          print("DELETE CUTIN LABEL")
          del self.labels[-1]
          self.newCutinTarget = None
          self.newTargetTimer = self.buffer
          self.cancelCutinTimer = self.cancelBuffer
          self.lastLabelProduced = self.labels[-1][0]




