from multiprocessing import Value, Lock
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from DataPoint import DataPoint
from Video import Video
import time

# https://elsampsa.github.io/valkka-examples/_build/html/qt_notes.html#python-multiprocessing
# https://stackoverflow.com/questions/15520957/python-multiprocessing-in-pyqt-application?rq=1

# https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Value
# https://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
class _ProgressTracker(object):
    def __init__(self):
      self.numFramesProcessed = Value('i', 0) # creates an int: https://docs.python.org/2/library/array.html#module-array
      self.totalNumFrames = Value('i', 0)
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

    def getProgress(self):
      with self.lock:
        self.hasAskedForProgress.value = True
        assert(self.totalNumFrames.value != 0)
        return self.numFramesProcessed.value / self.totalNumFrames.value
    
    def reset(self):
      with self.lock:
        self.numFramesProcessed.value = 0
        self.totalNumFrames.value = 0
        # find totalNumFrames first then the getProgress can be used
        self.hasAskedForProgress.value = False

def _processVideo(dataPoint: DataPoint):
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
    isFrameAvail, frame = video.getFrame()
    if not isFrameAvail:
      break

    # simulate doing some work
    frameIndex += 1
    if frameIndex % 75 == 0:
      labels += [dummyLabels[frameIndex//75-1]]

    vehicleDetector.getFeaturesForFrame(frame)

    progressTracker.incrementNumFramesProcessed()
  
  # return dummy data
  dataPoint.predictedLabels = labels
  dataPoint.laneLines = [(10,20,30,40),(50,60,70,80),(100,200,300,200)]

  return dataPoint

def _run(dataPoints, progressTracker, completedCallback, progressCallback, pool):
  assert(dataPoints != [])
  futures = set()
  for dataPoint in dataPoints:
    progressTracker.addToTotalNumFrames(dataPoint.videoPath)
    futures.add(pool.submit(_processVideo, (dataPoint)))
  while futures:
    done, notDone = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
    futures = notDone
    for future in done:
      completedCallback(future.result())
    percentDone = progressTracker.getProgress()
    progressCallback(percentDone)
    time.sleep(1)
  progressTracker.reset()

# TODO test if detectron uses multithreading when model is CPU. It should because it uses PyTorch.

# executed in the other python process
def loadLibs(progressTracker):
  from VehicleDetector import VehicleDetector
  from LaneLineDetector import LaneLineDetector
  globals()['vehicleDetector'] = VehicleDetector()
  globals()['laneLineDetector'] = LaneLineDetector()
  globals()['progressTracker'] = progressTracker

class Classifier:
  # These members cannot be globals in this file since multiprocessing forks this python
  # exe to create the new processes. So code in global scope would execute twice????????
  def __init__(self):
    self.progressTracker = _ProgressTracker()

    # only run one video at a time
    self.pool = ProcessPoolExecutor(max_workers=1, initializer=loadLibs, initargs=(self.progressTracker,))

  def processVideos(self, dataPoints, processingCompleteCallback, processingProgressCallback):
    from threading import Thread
    thread = Thread(target=_run, args=(dataPoints, self.progressTracker, processingCompleteCallback, processingProgressCallback, self.pool))
    thread.start()
