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

  def compareLabels(self):
    # For now just assign a random score
    self.aggregatePredConfidence = random.random()
    if False:
      nicepredictedlabels = self.predictedLabels
      nicegroundlabels = self.groundTruthLabels
      extralabels = []
      missinglabels = []
      i = 0   # predicted label counter
      j = 0   # ground truth counter
      found = 0.0
      extra = 0
      missing = 0
      wrongtime = 0
      while i < len(nicepredictedlabels):
        while j < len(nicegroundlabels):
          if nicepredictedlabels[i][0] == nicegroundlabels[j][0]:
            # label is correct and in correct slot
            difference = float(float(nicepredictedlabels[i][1]) - float(nicegroundlabels[j][1]))
            found += .5
            if abs(difference) <= 0.5:
              # label is correct and time is correct
              found += .5
            else:
              # label is correct but time is wrong
              wrongtime += 1
            i += 1
            j += 1
          else:
            # check the next labels to see if they match
            if nicepredictedlabels[i+1][0] == nicegroundlabels[j][0]:
              # we have an extra label
              tup = nicepredictedlabels[i][0], nicepredictedlabels[i][1]
              extralabels.append(tup)
              extra += 1
              i+=1
            elif nicepredictedlabels[i][0] == nicegroundlabels[j+1][0]:
              # we hav a missing label
              tup = nicegroundlabels[j][0], nicegroundlabels[j][1]
              missinglabels.append(tup)
              missing += 1
              j+=1
            else:
              # we are either missing multiple labels or have multiple additional labels
              i += 1
              j += 1

      total = len(nicegroundlabels)
      total = (found-(missing + extra))/total
      return total
