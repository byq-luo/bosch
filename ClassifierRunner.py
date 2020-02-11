from multiprocessing import Value, Lock
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from App import TESTING
if TESTING:
  from mock.Classifier import processVideo
  from mock.DataPoint import DataPoint
  from mock.Video import Video
else:
  from Classifier import processVideo
  from DataPoint import DataPoint
  from Video import Video
import time

# https://elsampsa.github.io/valkka-examples/_build/html/qt_notes.html#python-multiprocessing
# https://stackoverflow.com/questions/15520957/python-multiprocessing-in-pyqt-application?rq=1

# https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing.Value
# https://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
class _ProgressTracker(object):
  def __init__(self):
    self.numFramesProcessed = Value('i', 0) # creates an int: https://docs.python.org/2/library/array.html#module-array
    self.totalNumFrames = Value('i', 0)
    self.currentVideoProgress = Value('f', 0.0)
    self.hasAskedForProgress = Value('i', False)
    self.lock = Lock()

  def addToTotalNumFrames(self, videoPath: str):
    with self.lock:
      assert(not self.hasAskedForProgress.value)
      video = Video(videoPath)
      self.totalNumFrames.value += video.getTotalNumFrames()
      del video

  def incrementNumFramesProcessed(self):
    with self.lock:
      self.numFramesProcessed.value += 1
      assert(self.numFramesProcessed.value <= self.totalNumFrames.value)

  def setCurVidProgress(self, progress: float):
    with self.lock:
      self.currentVideoProgress.value = progress

  def getTotalProgress(self):
    with self.lock:
      self.hasAskedForProgress.value = True
      assert(self.totalNumFrames.value != 0)
      return self.numFramesProcessed.value / self.totalNumFrames.value

  def getCurVidProgress(self):
    with self.lock:
      self.hasAskedForProgress.value = True
      return self.currentVideoProgress.value

  def reset(self):
    with self.lock:
      self.numFramesProcessed.value = 0
      self.totalNumFrames.value = 0
      # find totalNumFrames first then the getProgress can be used
      self.hasAskedForProgress.value = False

# executed in the other python process
def _loadLibs(progressTracker, TESTING):
  if TESTING:
    # from VehicleDetectorDetectron import VehicleDetectorDetectron
    from mock.VehicleDetector import VehicleDetector
    from mock.LaneLineDetector import LaneLineDetector
  else:
    #from VehicleDetectorDetectron import VehicleDetectorDetectron as VehicleDetector
    from VehicleDetectorYolo import VehicleDetectorYolo as VehicleDetector
    from LaneLineDetector import LaneLineDetector

  globals()['vehicleDetector'] = VehicleDetector()
  globals()['laneLineDetector'] = LaneLineDetector()
  globals()['progressTracker'] = progressTracker

def _processVideo(dataPoint: DataPoint):
  # accesses the globals assigned above
  return processVideo(dataPoint, vehicleDetector, laneLineDetector, progressTracker)

def _run(dataPoints, progressTracker, completedCallback, progressCallback, pool):
  assert(dataPoints != [])
  for dataPoint in dataPoints:
    progressTracker.addToTotalNumFrames(dataPoint.videoPath)
  for dataPoint in dataPoints:
    future = pool.submit(_processVideo, (dataPoint))
    while not future.done():
      time.sleep(1)
      totalPercentDone = progressTracker.getTotalProgress()
      videoPercentDone = progressTracker.getCurVidProgress()
      progressCallback(totalPercentDone, videoPercentDone, dataPoint)
    completedCallback(future.result())
  progressTracker.reset()

class ClassifierRunner:
  # These members cannot be globals in this file since multiprocessing forks this python
  # exe to create the new processes? So code in global scope would execute twice????????
  def __init__(self):
    self.progressTracker = _ProgressTracker()

    # only run one video at a time
    self.pool = ProcessPoolExecutor(max_workers=1, initializer=_loadLibs, initargs=(self.progressTracker, TESTING))

  def processVideos(self, dataPoints, processingCompleteCallback, processingProgressCallback):
    from threading import Thread
    thread = Thread(target=_run, args=(dataPoints, self.progressTracker, processingCompleteCallback, processingProgressCallback, self.pool))
    thread.start()
