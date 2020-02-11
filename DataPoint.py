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
    #self.predictedLabels = []
    self.groundTruthLabels = []
    self.boundingBoxes = []
    self.segmentations = []
    self.laneLines = []
    self.aggregatePredConfidence = 0

    folder, nameExtension = os.path.split(videoPath)
    name, extension = os.path.splitext(nameExtension)
    self.videoName = name

    # try to load data from disk
    labelsFileName = name.replace('m0','labels.txt')
    labelFolder = folder.replace("video", "labels")
    try:
      with open(labelFolder + '/' + labelsFileName) as file:
        labelLines = [ln.rstrip('\n') for ln in file.readlines()]
        for ln in labelLines:
          lnList = ln.split(",")
          self.groundTruthLabels.append(lnList)

    except:
      self.groundTruthLabels = None

    try:
      dataFileName = name.replace('m0', 'data.pkl')
      with open(folder + '/' + dataFileName, 'rb') as file:
        data = pickle.load(file)
      print('DataPoint',videoPath,'loaded feature data')
    except:
      pass

    self.predictedLabels = [
      #["end", "1.7207001116268856"],      removed a predicted label
      ["evtEnd", "67.0"],
      ["rightTO=24", "90.52518677490954"],
      ["objTurnOff", "97.06598288913878"],
      ["evtEnd", "104.18015323449663"],
      ["end", "108.44646106956341"],
      ["evtEnd", "145.67672790279136"],
      ["rightTO=1", "149.09431953928677"],
      ["objTurnOff", "150.26606524322807"],
      ["added this label", "24857293294"],       #added a label on
      ["evtEnd", "151.73074737315466"],
      ["rightTO=2", "154.85540258366478"],
      ["lcRel", "172.7341986438553"],
      ["evtEnd", "189.80220021408857"],
      ["rightTO=10", "189.90754629604734"],
      ["lcRel", "190.15477445747212"],
      ["evtEnd", "196.7042976764161"],
      ["rightTO=18", "196.88222310522156"],
      ["cutout", "275.67011726722797"],
      ["lcRel", "275.9586137837218"],
      ["evtEnd", "280.0344186642869"],
      ["lcRel", "293.68984732511444"],
      ["cutin", "296.79716403023616"],
      ["evtEnd", "300.27941885114313"]]

    self.compareLabels()


  def saveToStorage(self, storage: Storage):
    pass

  def niceformat(self, labels):
    newlabels = []
    for label in labels:
      label1 = ""
      for char in label[0]:
        if char == "=":
          break
        else:
          label1+=char
      tup = label1, label[1]
      newlabels.append(tup)
    return newlabels

  def compareLabels(self):
    nicepredictedlabels = self.niceformat(self.predictedLabels)
    nicegroundlabels = self.niceformat(self.groundTruthLabels)
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



  '''
  def compareLabels(self):

    if self.groundTruthLabels is not None and self.predictedLabels is not None:
      missing = 0
      extra = 0
      total = len(self.groundTruthLabels)
      for gLabel in self.groundTruthLabels:
        found = False
        for pLabel in self.predictedLabels:
          if gLabel[0] == pLabel[0]:
            difference = float(gLabel[1]) - float(pLabel[1])
            if abs(difference) < 0.5:
              found = True
              break

        if not found:
          missing += 1

      for pLabel in self.predictedLabels:
        found = False
        for gLabel in self.groundTruthLabels:
          if pLabel[0] == gLabel[0]:
            difference = float(pLabel[1]) - float(gLabel[1])
            if abs(difference) < 0.5:
              found = True
              break
        if not found:
          extra += 1

      self.aggregatePredConfidence = (total - (missing + extra)) / total


      pass
    else:
      if random.random() < .75:
        self.aggregatePredConfidence = random.random()
    '''

