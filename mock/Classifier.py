from mock.DataPoint import DataPoint
from mock.Video import Video

def processVideo(dp: DataPoint,
                 vehicleDetector,
                 laneLineDetector,
                 progressTracker):
  video = Video(dp.videoPath)
  totalNumFrames = video.getTotalNumFrames()

  bboxes = []
  lines = []

  frameIndex = 0
  while True:
    isFrameAvail, frame = video.getFrame(vehicleDetector.wantsRGB)
    if not isFrameAvail:
      break

    vehicleBoxes = vehicleDetector.getBoxes(frame)

    laneLines = laneLineDetector.getLines(frame)

    bboxes.append(vehicleBoxes)
    lines.append(laneLines)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  labelsFileName = 'mock/predictedLabels.txt'
  with open(labelsFileName) as file:
    dp.predictedLabels = list(zip([ln.rstrip('\n') for ln in file.readlines()], range(1, 300, 10)))
  
  dp.boundingBoxes = bboxes
  dp.laneLines = lines

  return dp
