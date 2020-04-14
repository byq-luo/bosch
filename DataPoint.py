from Storage import Storage
import os, pickle, random

# This class is a relationship between a video and its data
class DataPoint:
  def __init__(self, videoPath: str):
    assert(videoPath != '')
    self.videoPath = videoPath
    self.initialLabelTime = None
    self.videoName = ''
    self.videoFolder = ''
    self.predictedLabels = []
    self.groundTruthLabels = []
    self.boundingBoxes = []
    self.laneLines = []
    self.aggregatePredConfidence = 0
    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name
    self.videoFolder = folder

    self.labelsPath = self.videoPath.replace('videos/', 'labels/').replace('m0.avi', 'labels.txt')
    self.featuresPath = self.videoPath.replace('videos/', 'labels/').replace('m0.avi', 'features.pkl')

    try:
      with open(self.featuresPath) as f:
        self.hasBeenProcessed = True
    except:
      self.hasBeenProcessed = False

    self._loadLabels()

  def _loadLabels(self):
    try: # load labels
      with open(self.labelsPath) as file:
        labelLines = [ln.rstrip('\n') for ln in file.readlines()]
        for ln in labelLines:
          label, labelTime = ln.split(',')
          label = label.split('=')[0]
          correctTime = float(labelTime) % 300
          self.groundTruthLabels.append((label, correctTime))
    except:
      self.groundTruthLabels = []

  # we cache the features on disk to avoid possible OOM when processing hundreds or thousands of videos
  def loadFeatures(self):
    try: # load features
      with open(self.featuresPath, 'rb') as file:
        [self.boundingBoxes, self.laneLines] = pickle.load(file)
    except:
      self.boundingBoxes = []
      self.laneLines = []

  def clearFeatures(self):
    self.boundingBoxes = []
    self.laneLines = []

  def labelsToOutputFormat(self, labels):
    ret = []
    for label,labelTime in labels:
      if label=='rightTO':
        label = label + '=XX'
      ret.append(label + ',' + str(labelTime) + '\n')
    return ret

  def saveToStorage(self, storage: Storage):
    labels = self.labelsToOutputFormat(self.predictedLabels)
    storage.writeListToFile(labels, self.labelsPath)
    storage.writeObjsToPkl([self.boundingBoxes,self.laneLines], self.featuresPath)
    self.boundingBoxes = []
    self.laneLines = []