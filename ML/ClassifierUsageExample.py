from BehaviorClassifier import BehaviorClassifier, ProgressTracker


##### To get progress use a thread and give the thread a callback. #####

# put callbacks somewhere
def processingProgressCallback(percent: float):
  print(percent)
def processingCompleteCallback(videoPath: str):
  print('Video',videoPath,'has completed processing.')

# pass callbacks to a thread that processes the videos
def wrapper(videosToProcess, behaviorClassifier, progressTracker, completedCallback, progressCallback):
  assert(videoPaths != [])
  videoFutures = []
  for videoPath in videosToProcess:
    videoFutures += [behaviorClassifier.processVideo.remote(videoPath)]
  completedVideos = []
  while videoFutures:
    completedVideo, videoFutures = ray.wait(videoFutures, timeout=.5)
    for videoPath in completedVideo:
      completedCallback(videoPath)
    completedVideos += completedVideo

    # We do not know how many frames all the videos have so we can't do something
    # like currentFrame / totalFrames
    currentVideoPercentComplete = ray.get(progressTracker.getProgress.remote())
    percentDone = len(completedVideos) / len(videosToProcess)
    progressCallback(percentDone)

    sleep(.5)

# ...

progressTracker = ProgressTracker.remote()
behaviorClassifier = BehaviorClassifier.remote()

from threading import Thread
thread = Thread(target=wrapper, args=(videoPaths, behaviorClassifier, progressTracker, processingCompleteCallback, processingProgressCallback))
thread.start()

