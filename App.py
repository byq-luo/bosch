from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QListWidgetItem
from PyQt5.QtCore import Qt
import PyQt5.QtCore as QtCore
from App_ui import Ui_MainWindow
import os

from ClassifierRunner import ClassifierRunner
from DataPoint import DataPoint
from Storage import Storage

# TODO TODO background workers do not stop if GUI is closed while processing
# TODO show parent folder of video in video list since some vids have same name?

DONT_PROCESS_VIDS = False

class MainWindow(QMainWindow):
  processingProgressSignal = QtCore.pyqtSignal(float)
  processingCompleteSignal = QtCore.pyqtSignal(DataPoint)

  def __init__(self):
    super(MainWindow, self).__init__()
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

    self.ui.playButton.clicked.connect(self.ui.videoWidget.play)
    self.ui.pauseButton.clicked.connect(self.ui.videoWidget.pause)
    self.ui.horizontalSlider.sliderMoved.connect(self.ui.videoWidget.seekToPercent)
    self.ui.processOneFileAction.triggered.connect(self.openFileNameDialog)
    self.ui.processMultipleFilesAction.triggered.connect(self.openFolderNameDialog)
    self.ui.videoWidget.setSlider(self.ui.horizontalSlider)
    self.ui.videoWidget.setTimeLabels(self.ui.currentVideoTime, self.ui.fullVideoTime)
    self.ui.boundingBoxCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawBoxes)
    self.ui.showLabelsCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawLabels)
    self.ui.fileListWidget.currentItemChanged.connect(self.videoInListClicked)
    self.ui.labelListWidget.currentItemChanged.connect(self.labelInListClicked)
    self.processingProgressSignal.connect(self.processingProgressUpdate)
    self.processingCompleteSignal.connect(self.processingComplete)

    # TODO what if user tries to process same video twice?
    self.dataPoints = dict()

    self.classifier = ClassifierRunner()

    # just a thin wrapper around a storage device
    self.storage = Storage()

  def labelInListClicked(self, item: QListWidgetItem, previousItem):
    if item is None:
      return
    frameIndex = item.data(Qt.UserRole)
    self.ui.videoWidget.seekToFrame(frameIndex)

  def setLabelList(self, dataPoint):
    self.ui.labelListWidget.clear()
    for label,frameIndex in dataPoint.predictedLabels:
      item = QListWidgetItem(label)
      item.setData(Qt.UserRole, frameIndex)
      self.ui.labelListWidget.addItem(item)

  def videoInListClicked(self, item: QListWidgetItem, previousItem):
    if item is None:
      return
    videoPath = item.data(Qt.UserRole)
    self.setCurrentVideo(self.dataPoints[videoPath])

  def setCurrentVideo(self, dataPoint, play=True):
    self.setLabelList(dataPoint)
    self.ui.videoWidget.setVideo(dataPoint)
    if play:
      self.ui.videoWidget.play()

  def addToVideoList(self, dataPoint):
    listItem = QListWidgetItem(dataPoint.videoName)
    listItem.setData(Qt.UserRole, dataPoint.videoPath)
    self.ui.fileListWidget.addItem(listItem)

  def openFileNameDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(self, caption="Choose a video",\
                                              filter="AVI/MKV Files (*.avi *.mkv)",\
                                              options=options)
    if fileName:
      dataPoint = DataPoint(fileName, self.storage)
      self.dataPoints[fileName] = dataPoint
      self.addToVideoList(dataPoint)

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
        dataPoint = DataPoint(videoPath, self.storage)
        self.dataPoints[dataPoint.videoPath] = dataPoint
        self.addToVideoList(dataPoint)
      if DONT_PROCESS_VIDS:
          return
      self.classifier.processVideos(
        self.dataPoints.values(),
        self.processingCompleteCallback,
        self.processingProgressCallback)

  def dummyKPI(self, dataPoint):
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

  # put the work onto the gui thread
  def processingProgressCallback(self, percent: float):
    self.processingProgressSignal.emit(percent)
  def processingProgressUpdate(self, percent: float):
    # update some widget or something
    print('Processing',percent,'complete.')

  def processingCompleteCallback(self, dataPoint: DataPoint):
    self.processingCompleteSignal.emit(dataPoint)
  def processingComplete(self, dataPoint: DataPoint):
    print('Video',dataPoint.videoPath,'has completed processing.')
    # id(oldVid) != id(dataPoint) so changes made to dataPoint in
    # BehaviorClassifier are not reflected in oldVid. oldVid and
    # dataPoint are different python objects.
    self.dataPoints[dataPoint.videoPath] = dataPoint
    self.dummyKPI(dataPoint)

    currentListItem = self.ui.fileListWidget.currentItem()
    if currentListItem is None:
      return
    currentVideoPath = currentListItem.data(Qt.UserRole)
    if currentVideoPath == dataPoint.videoPath:
      self.setCurrentVideo(dataPoint, play=False)


if __name__ == '__main__':
  import sys
  app = QApplication(sys.argv)
  window = MainWindow()
  window.show()
  sys.exit(app.exec_())
