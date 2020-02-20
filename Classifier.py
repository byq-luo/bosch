from LabelGenerator import LabelGenerator
# from VehicleTracker import VehicleTracker
# from VehicleTrackerDL import VehicleTracker
from VehicleTrackerSORT import VehicleTracker
from DataPoint import DataPoint
from Video import Video

from CONFIG import *

import numpy as np
import random
import cv2

# TODO somehow make use of frames from the future?
# TODO test if resizing the images makes performance (accuracy or speed) any better
# TODO make ERFNet work for any input image size
# TODO also make Detectron only give bounding boxes for vehicle classes
# TODO does ERFNet take BGR or RGB? what about DeepSort?
# TODO can get lane curves from prob map.
# See https://github.com/XingangPan/SCNN/tree/master/tools
# and https://github.com/cardwing/Codes-for-Lane-Detection


def _fillDataPoint(dp, rawBoxes, vehicles, envelopes, laneLines):
  boxes = []
  # Sending back min length box list works good
  vehicleBoxes = [v.box for v in vehicles]
  vehicleIDs = [v.id for v in vehicles]
  if len(vehicleBoxes) < len(rawBoxes):
    vehicleBoxes += [np.array([0, 0, 0, 0])] * (len(rawBoxes)-len(vehicleBoxes))
    vehicleIDs += ['.']
  if len(vehicleBoxes) > len(rawBoxes):
    rawBoxes += [np.array([0, 0, 0, 0])] * (len(vehicleBoxes)-len(rawBoxes))
  for box, vbox, _id in zip(rawBoxes, vehicleBoxes, vehicleIDs):
    boxes.append((list(map(int, box)), list(map(int, vbox)), _id))

  dp.boundingBoxes.append(boxes)
  dp.segmentations.append(envelopes)
  dp.laneLines.append(laneLines)


def processVideo(dp: DataPoint,
                 vehicleDetector,
                 laneLineDetector,
                 progressTracker,
                 TESTING):
  assert(not (TESTING and PRECOMPUTE))

  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()

  videoFeaturesPath = dp.videoPath.replace('videos', 'features').replace('.avi', '.pkl')
  if TESTING:
    vehicleDetector.loadFeaturesFromDisk(videoFeaturesPath)
    laneLineDetector.loadFeaturesFromDisk(videoFeaturesPath)

  tracker = VehicleTracker()
  labelGen = LabelGenerator(video.getFps())

  if PRECOMPUTE:
    allboxes = []
    allboxscores = []
    alllines = []
    allenvelopes = []
    allvehicles = []

  frames = []
  for frameIndex in range(totalNumFrames):
    if PRECOMPUTE:
      isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    if TESTING:
      isFrameAvail, frame = True, None
    if not isFrameAvail:
      print('Video='+dp.videoPath+' returned no frame for index=' +
            str(frameIndex)+' but totalNumFrames='+str(totalNumFrames))
      rawBoxes = []
      boxscores = []
      lines = []
      classes = []
      envelopes = []
      vehicles = []
    else:
      rawBoxes, envelopes, boxscores = vehicleDetector.getFeatures(frame)
      lines = laneLineDetector.getLines(frame)
      vehicles = tracker.getVehicles(frame, rawBoxes, boxscores)
      labelGen.processFrame(vehicles, lines, frameIndex)

    if PRECOMPUTE:
      allboxes.append(rawBoxes)
      allboxscores.append(boxscores)
      allenvelopes.append(envelopes)
      alllines.append(lines)
      allvehicles.append(vehicles)

    if not PRECOMPUTE:
      _fillDataPoint(dp, rawBoxes, vehicles, envelopes, lines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  if PRECOMPUTE:
    import pickle
    with open(videoFeaturesPath, 'wb') as file:
      pickle.dump([allboxes, allboxscores, allenvelopes, alllines, allvehicles], file)

  dp.predictedLabels = labelGen.getLabels()
  print(dp.predictedLabels)
  return dp