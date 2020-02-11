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
    self.videoFolder = ''
    self.predictedLabels = []
    self.groundTruthLabels = []
    self.boundingBoxes = []
    self.segmentations = []
    self.laneLines = []
    self.aggregatePredConfidence = 0

    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name
    self.videoFolder = folder

    self._loadFromStorage(storage)

  def _loadFromStorage(self, storage: Storage):
    labelsFileName = self.videoName.replace('m0', 'labels.txt')
    labelFolder = self.videoFolder.replace('video', 'labels')
    try:
      with open(labelFolder + '/' + labelsFileName) as file:
        labelLines = [ln.rstrip('\n') for ln in file.readlines()]
        for ln in labelLines:
          label, labelTime = ln.split(',')
          labelTime = float(labelTime)
          self.groundTruthLabels.append((label, labelTime))
    except:
      self.groundTruthLabels = None

  def saveToStorage(self, storage: Storage):
    pass

  def compareLabels(self):
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
