from Storage import Storage
import os, pickle, random
from Video import Video

# This class is a relationship between a video and its data
class DataPoint:
  def __init__(self, videoPath: str):
    assert(videoPath != '')
    self.videoPath = videoPath
    self.videoName = ''
    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name
    self.videoFileName = name + extension
    self.savePath = folder.replace('videos', 'labels')

    self.predictedLabels = [] # tuple(str, float)
    self.boundingBoxes = [] # tuple(something i do not remember)
    self.laneLines = [] # ..

    video = Video(videoPath)
    self.videoLength = video.getVideoLength()
    del video

    self.hasBeenProcessed = False
    self.setSavePath(self.savePath)
    if self.hasBeenProcessed:
      self._loadLabels()

  def _loadLabels(self):
    try:
      with open(self.labelsPath) as file:
        labelLines = [ln.rstrip('\n') for ln in file.readlines()]
        for ln in labelLines:
          label, labelTime = ln.split(',')
          label = label.split('=')[0] # handle rightTO label
          labelTime = float(labelTime) % 300
          self.predictedLabels.append((label, labelTime))
    except:
      self.predictedLabels = []

  # we cache the features on disk to avoid possible OOM when processing hundreds or thousands of videos
  def loadFeatures(self, storage):
    try:
      [self.boundingBoxes, self.laneLines] = storage.loadObjsFromPkl(self.featuresPath)
    except:
      self.clearFeatures()

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
    self.clearFeatures()
    self.predictedLabels = []
    self.hasBeenProcessed = False

  def saveToStorage(self, storage: Storage, shouldSaveFeatures):
    labels = self.labelsToOutputFormat(self.predictedLabels)
    storage.writeListToFile(labels, self.labelsPath)
    if shouldSaveFeatures:
      storage.writeObjsToPkl([self.boundingBoxes,self.laneLines], self.featuresPath)

    # TODO this may have wierd interaction with videowidget?
    self.boundingBoxes = []
    self.laneLines = []

  def setSavePath(self, folder):
    if len(folder) == 0:
      return
    if folder[-1] not in ['/','\\']:
      folder += '/'
    self.savePath = folder
    self.labelsPath = self.savePath + self.videoFileName.replace('m0.avi', 'labels.txt')
    self.featuresPath = self.savePath + self.videoFileName.replace('m0.avi', 'features.pkl')
    self.updateDone()

  def updateDone(self):
    try:
      with open(self.labelsPath) as f:
        self.hasBeenProcessed = True
    except:
      self.hasBeenProcessed = False
      self.predictedLabels = []