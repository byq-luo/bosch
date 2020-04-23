from VehicleTracker import Vehicle

def isLeft(a, b, c):
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0

class LabelGenerator:
  def __init__(self, videoFPS):

    self.currentTargetObject = None   # current target object
    self.newPotentialTarget = None    # The vehicle thats about to become the current target object
    self.newCutinTarget = None        # A vehicle cutting into the host vehicles lane
    self.lastLabelProduced = None     # The last label produced by the label generator
    self.targetDirection = None       # The direction the current TO is moving (used for cutout)
    self.lastTargetPos = None         # The last place the current TO was detected (used for cutout)
    self._time = None                 # The time in the video
    self.labels = []                  # The list of generated labels
    self.videoFPS = videoFPS          # The frames per second of the video (used to generate timestamp)

    self.lastTO = None                # The vehicle object of the last TO detected, is always None
                                      # unless the last TO disappeared in the center of the screen
                                      # because it was too far away

    self.lastTargetX = None           # The last x coordinate of the current TO
                                      # (used to determine objTurnOff)

    # These are general buffer variables used to reset event timers
    self.buffer = 30
    self.cancelBuffer = 15

    # This buffer and timer are used to determine if the target object has disappeared
    self.targetLostBuffer = 15
    self.targetLostTimer = self.targetLostBuffer

    # These timers are used to idenify a new target object
    self.newEventTimer = self.buffer
    self.cancelTimer = self.cancelBuffer

    # These timers are used to produce the cutin label
    self.newTargetTimer = self.buffer
    self.cancelCutinTimer = self.cancelBuffer

    # These timers are used to produce the cutout label
    self.cutoutTimer = self.buffer
    self.cancelCutoutTimer = self.cancelBuffer

    # These timers are used to cancel a false cutout label
    self.reverseCutoutTimer = self.buffer
    self.cancelReverseCutout = self.cancelBuffer

    # These timers are used to produce the lcRel label
    self.laneChangeTimer = self.buffer
    self.cancelLaneChangeTimer = self.buffer

    # These variables are used to determine if a host lane change has occured
    self.currentLeftX = 0
    self.currentRightX = 0
    self.prevLeftX = None
    self.prevRightX = None
    self.laneChangeBuffer = 100
    self.laneChangeDir = None
    self.laneChangeEndBuffer = 45
    self.laneChangeEndTimer = self.laneChangeEndBuffer

    # These variables are used to produce the end label
    self.endBuffer = 130
    self.endTimer = self.endBuffer
    self.cancelEndBuffer = 10
    self.cancelEndTimer = self.cancelEndBuffer

  '''
  This function returns the list of predicted labels
  '''
  def getLabels(self):
    return self.labels


  '''
  This function takes in the list of line segments for the left and right lane
  lines of the host vehicles lane. It then handles the production of the end label
  '''
  def checkForEndLabel(self, leftSegments, rightSegments):

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

  '''
  This function takes in two x values representing where the left and right
  lane lines intersect a pre-set y value. These x values are then used to 
  determine if one of the lane lines has jumped significantly along the x-axis
  indicating a host vehicle lane change
  '''
  def checkForHostLaneChange(self, firstLeftX, firstRightX):
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
        eventTime = self._time - (self.buffer / self.videoFPS)
        newLabel = ("evtEnd", eventTime)
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
  This function takes in a boolean that is true if the current target object
  is present in the list of vehicle objects for this frame. If false, it will reset
  the current target object to None if the target object was last detected in 
  the center of the screen or will produce a objTurnOff label if the target was
  last detected near the edges of the screen indicating that it turned off the 
  host vehicles road
  '''
  def checkForObjTurnOff(self, targetFound):

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
          self.labels.append(("evtEnd", self._time))
          self.lastLabelProduced = "evtEnd"
          self.lastTO = None

  '''
  This function takes in a list of vehicles in the hosts lane and returns
  the closest vehicle to the host
  '''
  def getClosestVehicleInLane(self, vehiclesInLane):
    y = 0
    closestTarget = None
    for vehicle in vehiclesInLane:
      # finds the closest target to the host vehicle
      if vehicle.box[3] > y:
        y = vehicle.box[3]
        closestTarget = vehicle
    return closestTarget


  '''
  This function takes in the closest Target to the host vehicle
  and is responsible for setting a new current target object
  '''
  def setCurrentTargetObject(self, closestTarget):
    targetInLane = False
    if closestTarget is not None:
      # Begin countdown making the closestTarget the new currentTarget
      if self.newPotentialTarget is None:
        self.label_time = self._time
        self.newPotentialTarget = closestTarget
        self.newEventTimer -= 1

      # Handles when a vehicle is becoming the new currentTarget
      elif self.newEventTimer > 0 and self.newEventTimer < self.buffer:
        if self.newPotentialTarget.id == closestTarget.id:
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
        self.lastTargetPos = "In Lane"
        targetInLane = True
        self.newPotentialTarget = None
        self.newEventTimer = self.buffer

    return targetInLane


  '''
  This function tracks whether or not the current target object is cutting
  out of the host vehicles lane. It takes in a boolean value thats true if
  the target is in the host lane and the lists of the vehicles on or outside
  of either lane line.
  '''
  def checkForCutout(self, targetInLane, vehiclesOutLaneLeft, vehiclesOnLeftLane,
                     vehiclesOnRightLane, vehiclesOutLaneRight):
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
        self.labels.append(newLabel)
        self.lastLabelProduced = "cutout"
        self.cutoutTimer = self.buffer
        self.cancelCutoutTimer = self.cancelBuffer
        if self.targetDirection == "Left":
          self.lastTargetPos = "On Left Line"
        if self.targetDirection == "Right":
          self.lastTargetPos = "On Right Line"


  '''
  This function is called when a cutout is in progress. It takes in the lists 
  of vehicles on or outside of each lane line as well as a boolean thats true 
  if the current target object is in the host vehicles lane. It then uses this
  information to either produce the evtEnd label at the completion of the cutout
  or erase the previous cutout label if the target returns to the hosts lane.
  '''
  def handleCutoutInProgress(self, targetInLane, vehiclesOutLaneLeft, vehiclesOnLeftLane,
                             vehiclesOnRightLane, vehiclesOutLaneRight):
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
        self.labels.append(newLabel)
        self.lastLabelProduced = "evtEnd"
        self.cutoutTimer = self.buffer
        self.cancelCutoutTimer = self.cancelBuffer
        self.currentTargetObject = None
        self.lastTargetPos = None
        self.targetDirection = None
        self.lastTO = None

    # Object in host lane
    elif targetInLane:
      if self.reverseCutoutTimer > 0:
        self.reverseCutoutTimer -= 1

      else:
        self.cutoutTimer = self.buffer
        self.reverseCutoutTimer = self.buffer
        self.cancelCutoutTimer = self.cancelBuffer
        self.cancelReverseCutout = self.cancelBuffer
        del self.labels[-1]
        self.lastLabelProduced = "rightTO"
        self.lastTargetPos = "In Lane"
        self.targetDirection = None

  '''
  This function takes in the list of vehicles in the host vehicles lane or on
  either of the lane lines. It then determines whether or not a vehicle is
  cutting in to the host lane and produces the cutin label as necessary
  '''
  def checkForCutin(self, vehiclesOnLeftLane, vehiclesInLane, vehiclesOnRightLane):
    closerSideTarget = None
    targetInAdjacentLane = False
    y = 0

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
          del self.labels[-1]
          self.newCutinTarget = None
          self.newTargetTimer = self.buffer
          self.cancelCutinTimer = self.cancelBuffer
          self.lastLabelProduced = self.labels[-1][0]


  '''
  This function is called for each frame of a video and takes in the list
  of vehicle objects, the list of line segments representing each lane line,
  and the frame index. It then uses this information as well as information
  stored from previous frames to predict labels associated with the video
  '''
  def processFrame(self, vehicles, lines, frameIndex):

    self._time = (frameIndex / self.videoFPS)

    if len(lines) != 2:
      return

    leftSegments = lines[0]
    rightSegments = lines[1]

    self.checkForEndLabel(leftSegments, rightSegments)
    if self.lastLabelProduced == "end":
      return

    '''
    This section sorts all of the vehicle objects into one of these lists based on 
    where the vehicle is in relation to the host vehicles lane
    '''
    vehiclesOutLaneLeft = []
    vehiclesOnLeftLane = []
    vehiclesInLane = []
    vehiclesOnRightLane = []
    vehiclesOutLaneRight = []

    targetFound = False

    # These two variables represent the x values of the lane lines at a certain
    # pre-set y value to determine if the lane lines have jumped significantly
    # along the x axis indicating a host vehicle lane change
    firstLeftX = None
    firstRightX = None

    for vehicle in vehicles:

      # This block of code gets the center x coordinate and bottom y coordinate
      # of the vehicle
      box = vehicle.box
      x1,y1,x2,y2 = box
      leftX = min(x1,x2)
      rightX = max(x1,x2)
      midX = (rightX + leftX) / 2
      bottomY = max(y1,y2)

      # This block sets the last x position of the current target object
      if self.currentTargetObject is not None:
        if vehicle.id == self.currentTargetObject.id:
          targetFound = True
          self.lastTargetX = midX

      # This block of code finds the left and right lane line segments that
      # exist at the same y-value as the bottom of the vehicle.
      # The lane line x values firstLeftX and firstRightX are also set here
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

      # This block uses the information gathered above to sort each vehicle
      # into the correct list
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

    '''
    Checks for a host lane change
    '''
    self.checkForHostLaneChange(firstLeftX, firstRightX)

    '''
    Ensures target still exists, produces objTurnOff if target disappeared off
    the side edges of the video
    '''
    if self.currentTargetObject is not None and self.lastLabelProduced != "cutin":
      self.checkForObjTurnOff(targetFound)

    '''
    Finds the closest vehicle in the host lane if it exists
    '''
    closestTarget = self.getClosestVehicleInLane(vehiclesInLane)

    '''
    Sets a target object if there is no current target object
    Otherwise checks that the current target object is still the 
    closest vehicle in lane
    '''
    targetInLane = False
    if self.currentTargetObject is None and self.lastLabelProduced != "cutin":
      targetInLane = self.setCurrentTargetObject(closestTarget)

    # Checks to see if currentTarget object is still in the lane
    elif closestTarget is not None and self.lastLabelProduced != "cutin":
      if self.currentTargetObject.id == closestTarget.id:
        targetInLane = True

    '''
    This section handles when a target object is in the host vehicle lane
    '''
    if self.lastLabelProduced == "rightTO" and self.currentTargetObject is not None:
      self.checkForCutout(targetInLane, vehiclesOutLaneLeft, vehiclesOnLeftLane,
                          vehiclesOnRightLane, vehiclesOutLaneRight)

    elif self.lastLabelProduced == "cutout":
      self.handleCutoutInProgress(targetInLane, vehiclesOutLaneLeft, vehiclesOnLeftLane,
                                  vehiclesOnRightLane, vehiclesOutLaneRight)

    '''
    This section will track and handle new cars cutting into our lane
    '''
    if self.lastLabelProduced == "lcRel" or self.lastLabelProduced == "cutout":
      return
    else:
      self.checkForCutin(vehiclesOnLeftLane, vehiclesInLane, vehiclesOnRightLane)






