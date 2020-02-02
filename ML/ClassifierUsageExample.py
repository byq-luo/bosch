from BehaviorClassifier import BehaviorClassifier, ProgressTracker

##### To get progress use a thread and give the thread a callback. #####

# pass callbacks to a thread that processes the videos
def wrapper(videosToProcess, completedCallback, progressCallback):
  assert(videoPaths != [])
  progressTracker = ProgressTracker.remote()
  behaviorClassifier = BehaviorClassifier.remote(progressTracker)
  videoFutures = []
  for videoPath in videosToProcess:
    progressTracker.addToTotalNumFrames.remote(videoPath)
    videoFutures += [behaviorClassifier.processVideo.remote(videoPath)]
  completedVideos = []
  while videoFutures:
    completedVideo, videoFutures = ray.wait(videoFutures, timeout=.5)
    for videoPath in completedVideo:
      completedCallback(videoPath)
    completedVideos += completedVideo

    currentVideoPercentComplete = ray.get(progressTracker.getProgress.remote())
    progressCallback(percentDone)

    sleep(.5)

# ...

def processingProgressCallback(percent: float):
  # Could use the slot for some Qt progress widget instead of this callback
  print('Processing',percent,'complete.')
def processingCompleteCallback(videoPath: str):
  print('Video',videoPath,'has completed processing.')

# ...

#### Note Only have one thread at a time ####
from threading import Thread
thread = Thread(target=wrapper, args=(videoPaths, processingCompleteCallback, processingProgressCallback))
thread.start()
