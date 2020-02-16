from VehicleTracker import VehicleTracker, Vehicle
from DataPoint import DataPoint
from Video import Video

import numpy as np
import random
import cv2

# TODO also make Detectron only give bounding boxes for vehicle classes

# Take what we want from the features
def _fillDataPoint(dp, vehicles, masksList, laneLinesNumpy):
  boxesList = []
  for vehicle in vehicles:
    box = list(vehicle.box)
    boxesList.append((list(map(int, box)), vehicle.id))
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

  for frameIndex in range(totalNumFrames):
    # isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    isFrameAvail, frame = True, None
    _time = frameIndex / videoFPS
    if not isFrameAvail:
      print('Video='+dp.videoPath+' returned no frame for index=' +
            str(frameIndex)+' but totalNumFrames='+str(totalNumFrames))
      boxes, masks = [], []
      lines = []
    else:
      # TODO test if resizing the images makes performance (accuracy or speed) any better
      # TODO make ERFNet work for any input image size
      # frame = frame[190:190+170,100:620].copy()
      boxes, masks = vehicleDetector.getFeatures(frame)
      #lines = laneLineDetector.getLines(frame)
      lines = []

      # TODO we can the equations from the LandLineDetector.
      if False: # for code folding in the editor
        # get lane line equations
        '''
              leftXB = laneLines[0][0]
              leftYB = laneLines[0][1]
              leftXT = laneLines[0][2]
              leftYT = laneLines[0][3]
              
              rightXB = laneLines[1][0]
              rightYB = laneLines[1][1]
              rightXT = laneLines[1][2]
              rightYT = laneLines[1][3]
              
              leftSlope = (leftYT - leftYB) / (leftXT - leftXB)
              leftInt = leftYB - (leftSlope * leftXB)
              
              rightSlope = (rightYT - rightYB) / (rightXT - rightXB)
              rightInt = rightYB - (rightSlope * rightXB)
              '''

        # This section finds all the boxes within the current lane
        # THINGS TO DO IN THIS SECTION:
        #     detect boxes that are half in the lane on left and right
        #     detect boxes completely out of lane on left and right
        '''
              boxesOutLaneLeft = []
              boxesOnLeftLane = []
              boxesInLane = []
              boxesOnRightLane = []
              boxesOutLaneRight = []
              
              for box in vehicleBoxes:
                insideLeftEdge = False
                insideRightEdge = False
                
                leftX = box[0]
                rightX = box[2]
                Y = box[3]
                
                leftLaneY = leftSlope*leftX + leftInt
                if Y > leftLaneY:
                  insideLeftEdge = True
                
                rightLaneY = rightSlope*rightX + rightInt
                if Y > rightLaneY:
                  insideRightEdge = True
                  
                if insideLeftEdge and insideRightEdge:
                  boxesInLane.append(box)
        '''

        # Actually produce the labels
        # THINGS TO DO IN THIS SECTION:
        #     Use ID of current Target object to find out which list its in:
        #     The first _time the target object leaves the lane, begin decreasing the new event timer
        #     If the new event timer reaches 0 and the target object has not returned to the lane, produce a new label
        #     Once the target object leaves the lane, start the new event timer
        #     When timer reaches 0 produce evtEnd label and set currentTargetObject to None
        '''
              # This section is used to determine the targetObject
              if currentTargetObject is None:
                y = 0
                targetFound = False
                for box in boxesInLane:
                  # finds the closest target to the host vehicle
                  if box[3] > y:
                    currentTargetObject = box
                    targetFound = True
                
                if targetFound:
                  newLabel = ("rightTO", _time)
                  labels.append(newLabel)
                  lastLabelProduced = "rightTO"

              '''

      vehicles = tracker.getObjs(frame, boxes)
      #vehicles = [Vehicle(0, b) for b in boxes]

    _fillDataPoint(dp, vehicles, masks, lines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  return dp


# For precomputing features
#allboxes, alllines, allmasks = [], [], []
# allboxes.append(boxes)
# allmasks.append(masks)
# alllines.append(lines)
# ...
#import pickle
# with open(videoFeaturesPath, 'wb') as file:
#  pickle.dump([allboxes, allmasks, alllines], file)
