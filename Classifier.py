from DataPoint import DataPoint
from VehicleDetector import VehicleDetector
from LaneLineDetector import LaneLineDetector
from Video import Video

# TODO this is also set in VehicleDetector
BBOX_SCORE_THRESH = .7

# Take what we want from the features
def _featuresToDataPoint(dp, vehicleFeatures, laneFeatures):
  # See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
  instances = vehicleFeatures['instances']
  boxes = instances.pred_boxes
  scores = instances.scores
  #classes = instances.pred_classes
  goodBoxes = boxes[scores > BBOX_SCORE_THRESH]
  boxes = []
  for boxTensor in goodBoxes:
    box = boxTensor.tolist()
    boxes.append(list(map(int,box)))
  dp.boundingBoxes.append(boxes)

  lanes = []
  for lane in laneFeatures:
    lanes.append(list(map(int, lane)))
  dp.laneLines.append(lanes)

def processVideo(dp: DataPoint,
                 vehicleDetector: VehicleDetector,
                 laneLineDetector: LaneLineDetector,
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
    isFrameAvail, frame = video.getFrame()
    if not isFrameAvail:
      break

    # simulate doing some work
    frameIndex += 1
    if frameIndex % 30 == 0:
      dp.predictedLabels.append((dummyLabels[(frameIndex//30-1)%len(dummyLabels)], frameIndex))

    vehicleFeatures = vehicleDetector.getFeatures(frame)
    laneFeatures = laneLineDetector.getLines(frame)

    _featuresToDataPoint(dp, vehicleFeatures, laneFeatures)
    progressTracker.setCurVidProgress(frameIndex / totalNumFrames)
    progressTracker.incrementNumFramesProcessed()

  return dp
