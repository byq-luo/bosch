from DataPoint import DataPoint
from Video import Video

# Take what we want from the features
def _featuresToDataPoint(dp, boxesTensor, masksList, laneLinesNumpy):
  boxesList = []
  for boxTensor in boxesTensor:
    box = boxTensor.tolist()
    boxesList.append(list(map(int,box)))
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

  videoFeaturesPath = dp.videoPath.replace('videos', 'features').replace('.avi', '.pkl')
  if TESTING:
    vehicleDetector.loadFeaturesFromDisk(videoFeaturesPath)
    laneLineDetector.loadFeaturesFromDisk(videoFeaturesPath)

  # TODO predict
  labels = []

  frameIndex = 0
  while True:
    isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    if not isFrameAvail:
      break
    frameIndex += 1

    vehicleBoxes, vehicleMasks = vehicleDetector.getFeatures(frame)
    laneLines = laneLineDetector.getLines(frame)

    # simulate doing some work
    fakeLabel = ('fake label ' + str(frameIndex), frameIndex / totalNumFrames)
    if frameIndex % 30 == 0:
      dp.predictedLabels.append(fakeLabel)

    _featuresToDataPoint(dp, vehicleBoxes, vehicleMasks, laneLines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()
  
  return dp


# For precomputing features
#boxes.append(vehicleBoxes)
#masks.append(vehicleMasks)
#lines.append(laneLines)
#...
#import pickle
#with open(videoFeaturesPath, 'wb') as file:
#  pickle.dump([boxes, masks, lines], file)