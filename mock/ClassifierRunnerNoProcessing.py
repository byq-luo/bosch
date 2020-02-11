import mock.Classifier
from threading import Thread
import time

def _run(dataPoints, completedCallback, progressCallback):
  assert(dataPoints != [])
  itersPerVid = 3
  totalIters = itersPerVid * len(dataPoints)

  for i, dataPoint in enumerate(dataPoints):
    iters = 1
    while iters <= itersPerVid:
      time.sleep(1)
      totalPercentDone = (i * itersPerVid + iters) / totalIters
      videoPercentDone = iters / itersPerVid
      progressCallback(totalPercentDone, videoPercentDone, dataPoint)
      iters += 1

    completedCallback(dataPoint)

class ClassifierRunner:
  def processVideos(self, dataPoints, processingCompleteCallback, processingProgressCallback):
    thread = Thread(target=_run, args=(dataPoints, processingCompleteCallback, processingProgressCallback), daemon=True)
    thread.start()
