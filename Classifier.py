from DataPoint import DataPoint
from Video import Video

import torch

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
                 progressTracker):
  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()

  dummyLabels = [
    'rightTO=XX,1',
    'cutin,2',
    'evtEnd,3',
    'rightTO=XX,4',
    'objTurnOff,5',
    'evtEnd,6',
    'rightTO=XX,7',
    'cutout,8',
    'evtEnd,9']

  frameIndex = 0
  while True:
    isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    if not isFrameAvail:
      break

    # simulate doing some work
    frameIndex += 1
    if frameIndex % 30 == 0:
      dp.predictedLabels.append((dummyLabels[(frameIndex//30-1)%len(dummyLabels)], frameIndex))

    vehicleBoxes, vehicleMasks = vehicleDetector.getBoxes(frame)
    laneLines = laneLineDetector.getLines(frame)

    _featuresToDataPoint(dp, vehicleBoxes, vehicleMasks, laneLines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  return dp
