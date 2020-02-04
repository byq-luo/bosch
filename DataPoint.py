from Storage import Storage
import os

# This class is a relationship between a video and its data
class DataPoint:
  # This constructor should throw on any failures.
  def __init__(self, videoPath: str, storage: Storage):
    assert(videoPath != '')
    self.videoPath = videoPath
    self.predictedLabels = []
    self.groundTruthLabels = []
    self.boundingBoxes = []
    self.laneLines = []

    # try to load data from disk
    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    name = name.replace('m0','labels')
    with open(folder + '/' + name + '.txt') as file:
      self.groundTruthLabels = [ln.rstrip('\n') for ln in file.readlines()]

    # ...
    pass

  def saveToStorage(self, storage: Storage):
    pass