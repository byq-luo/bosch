from LabelGenerator import LabelGenerator
# from VehicleTracker import VehicleTracker
# from VehicleTrackerDL import VehicleTracker
from VehicleTrackerSORT import VehicleTracker
from DataPoint import DataPoint
from Video import Video

import CONFIG

import numpy as np
import random
import cv2

# TODO make ERFNet work for any input image size
# TODO does ERFNet take BGR or RGB?
# See https://github.com/XingangPan/SCNN/tree/master/tools
# and https://github.com/cardwing/Codes-for-Lane-Detection


def _updateDataPoint(dp, rawboxes, vehicles, laneLines):
  boxes = []
  vehicleBoxes = [v.box for v in vehicles]
  vehicleIDs = [v.id for v in vehicles]

  # Make lists the same length by zero padding
  if len(vehicleBoxes) < len(rawboxes):
    vehicleBoxes += [np.array([0, 0, 0, 0])] * (len(rawboxes)-len(vehicleBoxes))
    vehicleIDs += ['.']
  if len(vehicleBoxes) > len(rawboxes):
    rawboxes += [np.array([0, 0, 0, 0])] * (len(vehicleBoxes)-len(rawboxes))

  for box, vbox, _id in zip(rawboxes, vehicleBoxes, vehicleIDs):
    boxes.append((list(map(int, box)),
                  list(map(int, vbox)),
                  _id))

  dp.boundingBoxes.append(boxes)
  dp.laneLines.append(laneLines)


def processVideo(dp: DataPoint,
                 vehicleDetector,
                 laneLineDetector,
                 progressTracker,
                 stopEvent):

  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()

  videoFeaturesPath = dp.videoPath.replace('videos', 'features').replace('.avi', '.pkl')
  if CONFIG.USE_PRECOMPUTED_FEATURES:
    vehicleDetector.loadFeaturesFromDisk(videoFeaturesPath)
    laneLineDetector.loadFeaturesFromDisk(videoFeaturesPath)

  tracker = VehicleTracker()
  labelGen = LabelGenerator(video.getFps())

  if CONFIG.MAKE_PRECOMPUTED_FEATURES:
    allboxes, allboxscores, allvehicles, alllines = [], [], [], []

  frames = []
  for frameIndex in range(totalNumFrames):

    if stopEvent.is_set():
      print("Classifier process exiting.",flush=True)
      return dp

    if CONFIG.SHOULD_LOAD_VID_FROM_DISK:
      isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    else:
      isFrameAvail, frame = True, None

    if not isFrameAvail:
      print('Video='+dp.videoPath+' returned no frame for index=' +
            str(frameIndex)+' but totalNumFrames='+str(totalNumFrames))
      rawboxes, boxscores, vehicles, lines = [], [], [], [[],[]]
    else:
      rawboxes, boxscores = vehicleDetector.getFeatures(frame)
      vehicles = tracker.getVehicles(frame, rawboxes, boxscores)
      lines = laneLineDetector.getLines(frame)
      labelGen.processFrame(vehicles, lines, frameIndex)

    if CONFIG.MAKE_PRECOMPUTED_FEATURES:
      allboxes.append(rawboxes)
      allboxscores.append(boxscores)
      allvehicles.append(vehicles)
      alllines.append(lines)

    _updateDataPoint(dp, rawboxes, vehicles, lines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  if CONFIG.MAKE_PRECOMPUTED_FEATURES:
    import pickle
    with open(videoFeaturesPath, 'wb') as file:
      pickle.dump([allboxes, allboxscores, alllines, allvehicles], file)

  dp.predictedLabels = labelGen.getLabels()
  dp.hasBeenProcessed = True
  return dp
