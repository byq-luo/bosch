from VehicleTracker import VehicleTracker
# from VehicleTrackerDL import VehicleTracker
# from VehicleTrackerSORT import VehicleTracker
from DataPoint import DataPoint
from Video import Video

import numpy as np
import random
import cv2

# TODO also make Detectron only give bounding boxes for vehicle classes

# Take what we want from the features
def _fillDataPoint(dp, boxes, nudgeboxes,ids, masksList, laneLinesNumpy):
  boxesList = []

  # Sending back min length box list works good
  if len(nudgeboxes) < len(boxes):
    nudgeboxes += [np.array([0,0,0,0])] * (len(boxes)-len(nudgeboxes))
    ids += ['.']
  if len(nudgeboxes) > len(boxes):
    boxes += [np.array([0,0,0,0])] * (len(nudgeboxes)-len(boxes))
  for box,nbox,_id in zip(boxes, nudgeboxes,ids):
    boxesList.append((list(map(int,box)), list(map(int, nbox)), _id))
  dp.boundingBoxes.append(boxesList)

  dp.segmentations.append(masksList)

  lanes = []
  for lane in laneLinesNumpy:
    lanes.append(list(map(int, lane)))
  dp.laneLines.append(lanes)


def processVideo(dp: DataPoint,
                 vehicleDetector,
                 laneLineDetector,
                 progressTracker,
                 TESTING):

  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()
  videoFPS = video.getFps()

  videoFeaturesPath = dp.videoPath.replace('videos', 'features').replace('.avi', '.pkl')
  if TESTING:
    vehicleDetector.loadFeaturesFromDisk(videoFeaturesPath)
    # laneLineDetector.loadFeaturesFromDisk(videoFeaturesPath)

  tracker = VehicleTracker()

  labels = []
  currentTargetObject = None
  newPotentialTarget = None
  newEventTimer = 10
  newTargetTimer = 10
  lastLabelProduced = None

  allboxes, allboxscores, alllines, allmasks = [], [], [], []

  for frameIndex in range(totalNumFrames):
    # isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    isFrameAvail, frame = True, None
    _time = frameIndex / videoFPS
    if not isFrameAvail:
      print('Video='+dp.videoPath+' returned no frame for index=' +
            str(frameIndex)+' but totalNumFrames='+str(totalNumFrames))
      boxes, boxscores, masks = [], [], []
      lines = []
    else:
      # TODO test if resizing the images makes performance (accuracy or speed) any better
      # TODO make ERFNet work for any input image size
      # frame = frame[190:190+170,100:620].copy()
      boxes, boxscores, masks = vehicleDetector.getFeatures(frame)
      #lines = laneLineDetector.getLines(frame)
      lines = []

      nudgeBoxes, ids = tracker.getObjs(frame, boxes, boxscores)
      # TODO we can the equations from the LandLineDetector.
      if len(lines) > 10000: # for code folding in the editor


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
              
        for box in nudgeBoxes:
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
            newLabel = ("rightTO", _time)
            labels.append(newLabel)
            lastLabelProduced = "rightTO"

        if lastLabelProduced == "rightTO":
          pass


        if lastLabelProduced == "objTurnOff":
          # Check how much time is left on new event timer
          # Also check that box is still on one of the lanes
          pass

        if lastLabelProduced == "evtEnd":
          pass



    _fillDataPoint(dp, boxes, nudgeBoxes, ids, masks, lines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()
  
  return dp


# For precomputing features
#allboxes, allboxscores, alllines, allmasks = [], [], [], []
#allboxes.append(boxes)
#allboxscores.append(boxscores)
#allmasks.append(masks)
#alllines.append(lines)
#import pickle
#with open(videoFeaturesPath, 'wb') as file:
#  pickle.dump([allboxes, allboxscores, allmasks, alllines], file)
