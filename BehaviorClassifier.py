from DataPoint import DataPoint

from Video import Video
import time
import ray
ray.init()


@ray.remote
class ProgressTracker:
  def __init__(self):
    self.reset()
  def addToTotalNumFrames(self, videoPath: str):
    assert(not self.hasAskedForProgress)
    video = Video(videoPath)
    self.totalNumFrames += video.get_total_num_frames()
    del video
  def incrementNumFramesProcessed(self):
    self.numFramesProcessed += 1
    assert(self.numFramesProcessed <= self.totalNumFrames)
  def getProgress(self):
    self.hasAskedForProgress = True
    assert(self.totalNumFrames != 0)
    return self.numFramesProcessed / self.totalNumFrames
  def reset(self):
    self.numFramesProcessed = 0
    self.totalNumFrames = 0
    # find totalNumFrames first then the getProgress can be used
    self.hasAskedForProgress = False


@ray.remote
class BehaviorClassifier:
  def __init__(self, progressTracker: ProgressTracker):
    import time
    import cv2
    from VehicleDetector import VehicleDetector
    from LaneLineDetector import LaneLineDetector
    self.vehicleDetector = VehicleDetector()
    self.laneLineDetector = LaneLineDetector()
    self.progressTracker = progressTracker

  def processVideo(self, dataPoint: DataPoint):
    video = Video(dataPoint.videoPath)

    labels = []

    dummyLabels = [
      'rightTO=XX,4',
      'objTurnOff,5',
      'evtEnd,6',
      'rightTO=XX,7',
      'evtEnd,9']

    frameIndex = 0
    while True:
      isFrameAvail, frame = video.get_frame()
      if not isFrameAvail:
        break


      # simulate doing some work
      time.sleep(.01)
      frameIndex += 1
      if frameIndex % 75 == 0:
        labels += [dummyLabels[frameIndex//75-1]]

      # vehicleDetector.getFeaturesForFrame(frame)

      self.progressTracker.incrementNumFramesProcessed.remote()
    
    # return the dummy labels
    dataPoint.predictedLabels = labels

    return dataPoint


def __run(dataPoints, behaviorClassifier, progressTracker, completedCallback, progressCallback):
  assert(dataPoints != [])
  futures = []
  for dataPoint in dataPoints:
    progressTracker.addToTotalNumFrames.remote(dataPoint.videoPath)
    futures += [behaviorClassifier.processVideo.remote(dataPoint)]
  while futures:
    result, futures = ray.wait(futures, timeout=1.0) # pass float to suppress a warning
    for dataPoint in result:
      completedCallback(ray.get(dataPoint))
    percentDone = ray.get(progressTracker.getProgress.remote())
    progressCallback(percentDone)
    time.sleep(1)
  ray.get(progressTracker.reset.remote())


def processVideos(dataPoints, behaviorClassifier, progressTracker, processingCompleteCallback, processingProgressCallback):
  from threading import Thread
  thread = Thread(target=__run, args=(dataPoints, behaviorClassifier, progressTracker, processingCompleteCallback, processingProgressCallback))
  thread.start()