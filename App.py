import sys
# sys.path.append('deepsort')

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem, QDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import PyQt5.QtCore as QtCore
from App_ui import Ui_MainWindow
import os

from StatsDialog import StatsDialog
from ClassifierRunner import ClassifierRunner
from DataPoint import DataPoint
from Storage import Storage

import CONFIG

# TODO TODO background workers do not stop if GUI is closed while processing

class MainWindow(QMainWindow):
  def __init__(self):
    super(MainWindow, self).__init__()
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)

    self.ui.playButton.clicked.connect(self.ui.videoWidget.play)
    self.ui.pauseButton.clicked.connect(self.ui.videoWidget.pause)
    self.ui.horizontalSlider.sliderMoved.connect(self.ui.videoWidget.seekToPercent)
    self.ui.videoWidget.setSlider(self.ui.horizontalSlider)
    self.ui.videoWidget.setTimeLabels(self.ui.currentVideoTime, self.ui.fullVideoTime)
    self.ui.boundingBoxCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawBoxes)
    self.ui.showLabelsCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawLabels)
    self.ui.showLaneLinesCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawLaneLines)
    # self.ui.showSegmentationsCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawSegmentations)
    self.ui.fileTableWidget.cellClicked.connect(self.videoInListClicked)
    self.ui.labelTableWidget.cellClicked.connect(self.labelInListClicked)
    self.ui.labelTableWidget.itemChanged.connect(self.labelInListChanged)
    self.processingProgressSignal.connect(self.processingProgressUpdate)
    self.processingCompleteSignal.connect(self.processingComplete)
    self.setWindowIcon(QIcon('icons/bosch.ico'))
    self.ui.actionInfo.triggered.connect(self.showInfoDialog)
    self.ui.actionProcessVideos.triggered.connect(self.initiateProcessing)

    # TODO what if user tries to process same video twice?
    self.dataPoints = dict()

    self.classifier = ClassifierRunner()

    # just a thin wrapper around a storage device
    self.storage = Storage()

    # If we are in TESTING mode just load videos from the precomputed folder
    if CONFIG.TESTING:
      self.loadVideosFromFolder('precomputed/videos')
    else:
      self.ui.processMultipleFilesAction.triggered.connect(self.openFolderNameDialog)

    self.dialog = StatsDialog()

  def showInfoDialog(self):
    self.dialog.show()

  def labelInListClicked(self, row, column):
    videoPath, labelIndex = self.ui.labelTableWidget.currentItem().data(Qt.UserRole)
    dp : DataPoint = self.dataPoints[videoPath]
    frameIndex = dp.predictedLabels[labelIndex][1]
    self.ui.videoWidget.seekToTime(frameIndex)

  def labelInListChanged(self, newItem):
    videoPath, index = newItem.data(Qt.UserRole)
    dataPoint = self.dataPoints[videoPath]
    try:
      newLabel = newItem.text()
      assert newLabel.find(" ") != -1
      label, time = newLabel.split(" ")
      time = time.strip()
      label = label.strip()
      assert label.isalpha() is True
      finalLabel = (label, float(time))
      dataPoint.predictedLabels[index] = finalLabel
      # causes enters inf loop due to signals
      # newItem.setText('{:10s} {}'.format(label, time))
    except:
      label, time = dataPoint.predictedLabels[index]
      # newItem.setText('{:10s} {}'.format(label, time))

  def setLabelList(self, dataPoint):
    self.ui.labelTableWidget.setRowCount(0)
    listIndex = 0
    videoPath = dataPoint.videoPath
    for label, labelTime in dataPoint.predictedLabels:
      rowIndex = self.ui.labelTableWidget.rowCount()
      self.ui.labelTableWidget.insertRow(rowIndex)
      item = QTableWidgetItem(' {:10s} {:.1f}'.format(label, labelTime))
      # TODO provide a more useful time measure
      #(either use the following uncommented line or change the time representation in the video widget)
      #item = QTableWidgetItem('{:10s} {:02d}:{:02d}'.format(label, labelTime//60, labelTime%60))
      item.setData(Qt.UserRole, (videoPath, listIndex))
      self.ui.labelTableWidget.setItem(rowIndex, 0, item)
      listIndex += 1

  def videoInListClicked(self, row, column):
    videoPath = self.ui.fileTableWidget.currentItem().data(Qt.UserRole)
    self.setCurrentVideo(self.dataPoints[videoPath])

  def setCurrentVideo(self, dataPoint, play=True):
    self.setLabelList(dataPoint)
    self.ui.videoWidget.setVideo(dataPoint)
    if play:
      self.ui.videoWidget.play()

  def addToVideoList(self, dataPoint: DataPoint):
    rowIndex = self.ui.fileTableWidget.rowCount()
    self.ui.fileTableWidget.insertRow(rowIndex)
    # TODO
    msg = ' ?'
    if dataPoint.aggregatePredConfidence != 0:
      msg = ' {:2.2f}'.format(dataPoint.aggregatePredConfidence)
    name = QTableWidgetItem(dataPoint.videoName)
    name.setData(Qt.UserRole, dataPoint.videoPath)
    score = QTableWidgetItem(msg)
    score.setData(Qt.UserRole, dataPoint.videoPath)
    self.ui.fileTableWidget.setItem(rowIndex, 0, score)
    self.ui.fileTableWidget.setItem(rowIndex, 1, name)

  def loadVideosFromFolder(self, folder):
    videoPaths = self.storage.recursivelyFindVideosInFolder(folder)
    for videoPath in videoPaths:
      if CONFIG.TESTING: # Do not load videos that have no precomputed boxes in TESTING mode
        videoFeaturesPath = videoPath.replace('videos/', 'features/').replace('.avi', '.pkl')
        if not os.path.isfile(videoFeaturesPath):
          continue
      dataPoint = DataPoint(videoPath)
      self.dataPoints[dataPoint.videoPath] = dataPoint
      self.addToVideoList(dataPoint)

  def initiateProcessing(self):
    self.classifier.processVideos(
      list(self.dataPoints.values()),
      self.processingCompleteCallback,
      self.processingProgressCallback)

  def openFileNameDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(self, caption="Choose a video",\
                                              filter="AVI/MKV Files (*.avi *.mkv)",\
                                              options=options)
    if fileName:
      dataPoint = DataPoint(fileName)
      self.dataPoints[dataPoint.videoPath] = dataPoint
      self.addToVideoList(dataPoint)

  def openFolderNameDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.ShowDirsOnly
    folderName = QFileDialog.getExistingDirectory(self, caption="Select Directory", options=options)
    if folderName:
      self.loadVideosFromFolder(folderName)

  # put the work onto the gui thread
  processingProgressSignal = QtCore.pyqtSignal(float, float, DataPoint)
  def processingProgressCallback(self, totalPercentDone: float, currentPercentDone: float, dataPoint: DataPoint):
    self.processingProgressSignal.emit(totalPercentDone, currentPercentDone, dataPoint)
  def processingProgressUpdate(self, totalPercentDone: float, currentPercentDone: float, dataPoint: DataPoint):
    msg = 'Total : {:3d}%   |   Current : {:3d}%   |   Video : {}'.format(int(totalPercentDone*100),int(currentPercentDone*100),dataPoint.videoPath)
    self.ui.statusbar.showMessage(msg, 3000)

  processingCompleteSignal = QtCore.pyqtSignal(DataPoint)
  def processingCompleteCallback(self, dataPoint: DataPoint):
    self.processingCompleteSignal.emit(dataPoint)
  def processingComplete(self, dataPoint: DataPoint):
    print('Video',dataPoint.videoPath,'has completed processing.')
    # id(oldVid) != id(dataPoint) so changes made to dataPoint in
    # BehaviorClassifier are not reflected in oldVid. oldVid and
    # dataPoint are different python objects.

    dataPoint.compareLabels()
    # self.videoScoreChanged(dataPoint)

    dataPoint.saveToStorage(self.storage)

    self.dataPoints[dataPoint.videoPath] = dataPoint

    currentItem = self.ui.videoWidget.dataPoint
    if currentItem is not None:
      currentVideoPath = self.ui.videoWidget.dataPoint.videoPath
      if currentVideoPath == dataPoint.videoPath:
        self.setCurrentVideo(dataPoint, play=False)

if __name__ == '__main__':
  # this solves a gross bug in cv2.cvtColor on macOS
  # See https://github.com/opencv/opencv/issues/5150
  import multiprocessing as mp
  mp.set_start_method('spawn')


  app = QApplication(sys.argv)
  window = MainWindow()
  window.show()
  sys.exit(app.exec_())
