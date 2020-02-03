from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QListWidgetItem
from App_ui import Ui_MainWindow
import os

from BehaviorClassifier import BehaviorClassifier, ProgressTracker, processVideos
from DataPoint import DataPoint
from Storage import Storage

# TODO TODO background workers do not stop if GUI is closed while processing

# TODO do we want to process the videos as soon as they're selected from the dialog?
# TODO there is a warning given by ray that we might want to checkout eventually

class MainWindow(QMainWindow):
  def __init__(self):
    super(MainWindow, self).__init__()
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

    self.ui.playButton.clicked.connect(self.ui.videoWidget.play)
    self.ui.pauseButton.clicked.connect(self.ui.videoWidget.pause)
    self.ui.horizontalSlider.sliderMoved.connect(self.ui.videoWidget.seekToPercent)
    self.ui.processOneFileAction.triggered.connect(self.openFileNameDialog)
    self.ui.processMultipleFilesAction.triggered.connect(self.openFolderNameDialog)
    self.ui.fileListWidget.currentTextChanged.connect(self.videoInListClicked)
    self.ui.videoWidget.setSlider(self.ui.horizontalSlider)

    # TODO what if user tries to process same video twice?
    self.dataPoints = dict()

    self.progressTracker = ProgressTracker.remote()
    self.behaviorClassifier = BehaviorClassifier.remote(self.progressTracker)

    # just a thin wrapper around a storage device
    self.storage = Storage()
  
  def videoInListClicked(self, videoPath: str):
    self.ui.labelListWidget.clear()
    dataPoint = self.dataPoints[videoPath]
    for label in dataPoint.predictedLabels:
      self.ui.labelListWidget.addItem(QListWidgetItem(label))
    self.ui.videoWidget.setVideoPath(videoPath)

  def openFileNameDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(self, caption="Choose a video",\
                                              filter="AVI/MKV Files (*.avi *.mkv)",\
                                              options=options)
    if fileName:
      self.ui.fileListWidget.addItem(QListWidgetItem(fileName))

  def openFolderNameDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.ShowDirsOnly
    folderName = QFileDialog.getExistingDirectory(self, caption="Select Directory",\
                                                  options=options)
    if folderName:
      videoPaths = self.storage.recursivelyFindVideosInFolder(folderName)
      # TODO this just blindly processes videos for now
      for videoPath in videoPaths:
        try:
          self.dataPoints[videoPath] = DataPoint(videoPath, self.storage)
          self.ui.fileListWidget.addItem(QListWidgetItem(dataPoint.videoPath))
        except: # If DataPoint fails to construct just skip this video
          pass
      processVideos(
        self.dataPoints.values(),
        self.behaviorClassifier,
        self.progressTracker,
        self.processingCompleteCallback,
        self.processingProgressCallback)

  def processingProgressCallback(self, percent: float):
    # update some widget or something
    print('Processing',percent,'complete.')

  def processingCompleteCallback(self, dataPoint: DataPoint):
    print('Video',dataPoint.videoPath,'has completed processing.')
    # id(oldVid) != id(dataPoint) so changes made to dataPoint in
    # BehaviorClassifier are not reflected in oldVid. oldVid and
    # dataPoint are different python objects.
    self.dataPoints[dataPoint.videoPath] = dataPoint

    # TODO the KPI computation should not be in the GUI code here

    # TODO just to get KPIs lets compare against groundtruth here
    print()
    print(dataPoint.videoPath)
    trueLabels = dataPoint.groundTruthLabels
    predLabels = dataPoint.predictedLabels
    print(trueLabels)
    print(predLabels)
    numerator = 0.0
    denominator = len(trueLabels) + abs(len(trueLabels) - len(predLabels))
    for trueLabel, predLabel in zip(trueLabels, predLabels):
      if trueLabel[:5] == predLabel[:5]:
        numerator += 1.0
    accuracy = numerator / denominator
    print('Accuracy', accuracy)
    print()


import sys
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
