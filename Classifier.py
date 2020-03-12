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

# TODO somehow make use of frames from the future?
# TODO test if resizing the images makes performance (accuracy or speed) any better
# TODO make ERFNet work for any input image size
# TODO also make Detectron only give bounding boxes for vehicle classes
# TODO does ERFNet take BGR or RGB? what about DeepSort?
# See https://github.com/XingangPan/SCNN/tree/master/tools
# and https://github.com/cardwing/Codes-for-Lane-Detection


def _updateDataPoint(dp, rawboxes, vehicles, laneLines):
  boxes = []
  # Sending back min length box list works good
  vehicleBoxes = [v.box for v in vehicles]
  vehicleIDs = [v.id for v in vehicles]
  if len(vehicleBoxes) < len(rawboxes):
    vehicleBoxes += [np.array([0, 0, 0, 0])] * (len(rawboxes)-len(vehicleBoxes))
    vehicleIDs += ['.']
  if len(vehicleBoxes) > len(rawboxes):
    rawboxes += [np.array([0, 0, 0, 0])] * (len(vehicleBoxes)-len(rawboxes))
  for box, vbox, _id in zip(rawboxes, vehicleBoxes, vehicleIDs):
    boxes.append((list(map(int, box)), list(map(int, vbox)), _id))

  dp.boundingBoxes.append(boxes)
  dp.laneLines.append(laneLines)


def processVideo(dp: DataPoint,
                 vehicleDetector,
                 laneLineDetector,
                 progressTracker):

  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()

  videoFeaturesPath = dp.videoPath.replace('videos', 'features').replace('.avi', '.pkl')
  if CONFIG.USE_PRECOMPUTED_FEATURES:
    vehicleDetector.loadFeaturesFromDisk(videoFeaturesPath)
    laneLineDetector.loadFeaturesFromDisk(videoFeaturesPath)

  tracker = VehicleTracker()
  labelGen = LabelGenerator(video.getFps())

  if CONFIG.MAKE_PRECOMPUTED_FEATURES:
    allboxes, allboxscores, allvehicles, alllines, alllanescores, allboxcornerprobs = [], [], [], [], [], []

  frames = []
  for frameIndex in range(totalNumFrames):
    if CONFIG.SHOULD_LOAD_VID_FROM_DISK:
      isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    else:
      isFrameAvail, frame = True, None

    if not isFrameAvail:
      print('Video='+dp.videoPath+' returned no frame for index=' +
            str(frameIndex)+' but totalNumFrames='+str(totalNumFrames))
      rawboxes, boxscores, vehicles, lines, lanescores, boxcornerprobs = [], [], [], [], [], []
    else:
      rawboxes, boxscores = vehicleDetector.getFeatures(frame)
      vehicles = tracker.getVehicles(frame, rawboxes, boxscores)
      lines, lanescores, boxcornerprobs = laneLineDetector.getLines(frame, vehicles)
      try: # TODO adam your code still has a bug
        labelGen.processFrame(vehicles, lines, frameIndex)
      except:
        pass

    if CONFIG.MAKE_PRECOMPUTED_FEATURES:
      allboxes.append(rawboxes)
      allboxscores.append(boxscores)
      allvehicles.append(list([v.id,*v.box] for v in vehicles))
      alllines.append(lines)
      alllanescores.append(lanescores)
      allboxcornerprobs.append(boxcornerprobs)

    _updateDataPoint(dp, rawboxes, vehicles, lines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  if CONFIG.MAKE_PRECOMPUTED_FEATURES:
    import pickle
    with open(videoFeaturesPath, 'wb') as file:
      pickle.dump([allboxes, allboxscores, alllines, alllanescores, allvehicles, allboxcornerprobs], file)

  dp.predictedLabels = labelGen.getLabels()
  return dp
