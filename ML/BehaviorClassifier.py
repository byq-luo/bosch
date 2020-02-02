import ray
ray.init()


@ray.remote
class ProgressTracker:
  def __init__(self):
    self.numFramesProcessed = 0
    self.totalNumFrames = 0
    # find totalNumFrames first then the getProgress can be used
    self.hasAskedForProgress = False
  def addToTotalNumFrames(self, videoPath):
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

  def processVideo(self, videoPath: str):
    # TODO do we have to put the import here when using ray?
    from Video import Video
    video = Video(videoPath)


    while True:
      isFrameAvail, frame = video.get_frame()

      # do work on frame
      time.sleep(.5)

      # vehicleDetector.getFeaturesForFrame(frame)

      self.progressTracker.incrementNumFramesProcessed.remote()

    return videoPath

