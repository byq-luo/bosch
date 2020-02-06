from DataPoint import DataPoint
from VehicleDetector import VehicleDetector
from LaneLineDetector import LaneLineDetector
from Video import Video

# TODO this is also set in VehicleDetector
BBOX_SCORE_THRES = .7

def processVideo(dataPoint: DataPoint,
                 vehicleDetector: VehicleDetector,
                 laneLineDetector: LaneLineDetector,
                 progressTracker):
  video = Video(dataPoint.videoPath)

  labels = []

  dummyLabels = [
    'rightTO=XX,4',
    'objTurnOff,5',
    'evtEnd,6',
    'rightTO=XX,7',
    'evtEnd,9']

  keepBoxes = []

  frameIndex = 0
  while True:
    isFrameAvail, frame = video.getFrame()
    if not isFrameAvail:
      break

    # simulate doing some work
    frameIndex += 1
    if frameIndex % 75 == 0:
      labels += [dummyLabels[frameIndex//75-1]]

    # See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    output = vehicleDetector.getFeaturesForFrame(frame)
    instances = output['instances']
    boxes = instances.pred_boxes
    scores = instances.scores
    #classes = instances.pred_classes
    keepBoxes.append([list(map(int,b.tolist())) for b in boxes[scores > BBOX_SCORE_THRES]])

    progressTracker.incrementNumFramesProcessed()

  # return dummy data
  dataPoint.predictedLabels = labels
  dataPoint.laneLines = [(10,20,30,40),(50,60,70,80),(100,200,300,200)]
  dataPoint.boundingBoxes = keepBoxes

  return dataPoint
