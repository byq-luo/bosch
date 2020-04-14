from Storage import Storage
import os, pickle, random
from Video import Video

# This class is a relationship between a video and its data
class DataPoint:
  def __init__(self, videoPath: str):
    assert(videoPath != '')
    self.videoPath = videoPath
    self.initialLabelTime = None
    self.videoName = ''
    self.videoFolder = ''
    self.predictedLabels = []
    self.boundingBoxes = []
    self.laneLines = []
    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name
    self.videoFolder = folder

    video = Video(videoPath)
    self.videoLength = video.getVideoLength()
    del video

    self.labelsPath = self.videoPath.replace('videos/', 'labels/').replace('m0.avi', 'labels.txt')
    self.featuresPath = self.videoPath.replace('videos/', 'labels/').replace('m0.avi', 'features.pkl')

    try:
      with open(self.featuresPath) as f:
        self.hasBeenProcessed = True
        self._loadLabels()
    except:
      self.hasBeenProcessed = False

  def _loadLabels(self):
    try: # load labels
      with open(self.labelsPath) as file:
        labelLines = [ln.rstrip('\n') for ln in file.readlines()]
        for ln in labelLines:
          label, labelTime = ln.split(',')
          label = label.split('=')[0]
          correctTime = float(labelTime) % 300
          self.predictedLabels.append((label, correctTime))
    except:
      self.predictedLabels = []

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
  
  def deleteData(self):
    try:
      os.remove(self.featuresPath)
    except:
      pass
    try:
      os.remove(self.labelsPath)
    except:
      pass
    self.hasBeenProcessed=False

  def saveToStorage(self, storage: Storage):
    labels = self.labelsToOutputFormat(self.predictedLabels)
    storage.writeListToFile(labels, self.labelsPath)
    storage.writeObjsToPkl([self.boundingBoxes,self.laneLines], self.featuresPath)
    self.boundingBoxes = []
    self.laneLines = []