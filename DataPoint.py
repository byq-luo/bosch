from Storage import Storage
import os, pickle, random
from Video import Video

# This class is a relationship between a video and its data
class DataPoint:
  def __init__(self, videoPath: str, storage):
    assert(videoPath != '')
    self.videoPath = videoPath
    self.videoName = ''
    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name
    self.videoFileName = name + extension
    self.savePath = folder

    self.predictedLabels = [] # tuple(str, float)
    self.boundingBoxes = [] # tuple(something i do not remember)
    self.laneLines = [] # ..

    video = Video(videoPath)
    self.videoLength = video.getVideoLength()
    del video

    self.hasBeenProcessed = False
    self.setSavePath(self.savePath, storage)
    if self.hasBeenProcessed:
      self._loadLabels(storage)

  def _loadLabels(self, storage):
    try:
      lines = storage.getFileLines(self.labelsPath)
      for ln in lines:
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

  def deleteData(self, storage):
    try:
      storage.deleteFile(self.featuresPath)
    except:
      pass
    try:
      storage.deleteFile(self.labelsPath)
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

  def setSavePath(self, folder, storage):
    if len(folder) == 0:
      return
    if folder[-1] not in ['/','\\']:
      folder += '/'
    self.savePath = folder
    self.labelsPath = self.savePath + self.videoFileName.replace('m0.avi', 'labels.txt')
    self.featuresPath = self.savePath + self.videoFileName.replace('m0.avi', 'features.pkl')
    self.updateDone(storage)

  def updateDone(self, storage):
    self.hasBeenProcessed = storage.fileExists(self.labelsPath)
    if not self.hasBeenProcessed:
      self.predictedLabels = []