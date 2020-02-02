import time
import ray
ray.init()


@ray.remote
class ProgressTracker:
  def __init__(self):
    self.progress = 0.0
  def setProgress(self, progress):
    self.progress = progress
  def getProgress(self):
    return self.progress


@ray.remote
class BehaviorClassifier:
  def __init__(self, progressTracker: ProgressTracker):
    import cv2
    from VehicleDetector import VehicleDetector
    from LaneLineDetector import LaneLineDetector
    self.vehicleDetector = VehicleDetector()
    self.laneLineDetector = LaneLineDetector()

  def processVideo(self, videoPath: str):
    # do fake work
    time.sleep(.5)

    # # call a function on the object
    # rayFuncCallId = vehicleDetector.getFeaturesForFrame.remote(frame)
    # # get the return value from function 
    # returnValue = ray.get(rayFuncCallId)

    # video = cv2.VideoCapture(videoPath)
    # ...

    return videoPath

