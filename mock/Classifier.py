from mock.DataPoint import DataPoint
from mock.Video import Video
import pickle

def processVideo(dp: DataPoint,
                 vehicleDetector,
                 laneLineDetector,
                 progressTracker):
  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()

  bboxes = []
  lines = []
  segmentations = []

  frameIndex = 0
  while True:
    isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    if not isFrameAvail:
      break

    vehicleBoxes, vehicleSegmentations = vehicleDetector.getBoxes(frame)
    laneLines = laneLineDetector.getLines(frame)

    # DO LABEL GENERATION HERE

    bboxes.append(vehicleBoxes)
    lines.append(laneLines)
    segmentations.append(vehicleSegmentations)

    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  with open('mock/predictedLabels.txt') as file:
    dp.predictedLabels = list(zip([ln.rstrip('\n') for ln in file.readlines()], range(1, 300, 10)))

  dp.boundingBoxes = bboxes
  dp.laneLines = lines
  dp.segmentations = segmentations

  return dp
