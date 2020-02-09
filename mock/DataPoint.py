import os, random
from mock.Storage import Storage

class DataPoint:
  def __init__(self, videoPath: str, storage: Storage):
    assert(videoPath != '')
    self.videoPath = ''
    self.videoName = ''
    self.predictedLabels = []
    self.groundTruthLabels = []
    self.boundingBoxes = []
    self.laneLines = []
    self.aggregatePredConfidence = 0

    folder, nameExtension = os.path.split(videoPath)

    # provide fake name
    folder = 'C:/Datasets/DashCamVideos/Gen5/'
    self.videoPath = folder + nameExtension

    name, extension = os.path.splitext(nameExtension)
    self.videoName = name

    if random.random() < .75:
      self.aggregatePredConfidence = random.random()

  def saveToStorage(self, storage: Storage):
    pass

