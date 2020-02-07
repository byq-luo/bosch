from Storage import Storage
import pickle
import os

import random

# This class is a relationship between a video and its data
class DataPoint:
  def __init__(self, videoPath: str, storage: Storage):
    assert(videoPath != '')
    self.videoPath = videoPath
    self.videoName = ''
    self.predictedLabels = []
    self.groundTruthLabels = []
    self.boundingBoxes = []
    self.laneLines = []
    self.aggregatePredConfidence = 0

    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name

    # try to load data from disk
    labelsFileName = name.replace('m0','labels.txt')
    labelFolder = folder.replace('video', 'labels')

    try:
      with open(folder + '/' + labelsFileName) as file:
      #with open(labelFolder + '/' + labelsFileName) as file:
        self.groundTruthLabels = [ln.rstrip('\n') for ln in file.readlines()]

    except:
      self.groundTruthLabels = None

    try:
      dataFileName = name.replace('m0', 'data.pkl')
      with open(folder + '/' + dataFileName, 'rb') as file:
        data = pickle.load(file)
      print('DataPoint',videoPath,'loaded feature data')
    except:
      pass

    # TODO
    self.aggregatePredConfidence = random.random()

    # ...
    pass

  def saveToStorage(self, storage: Storage):
    pass

