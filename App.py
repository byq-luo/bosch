from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem, QDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import PyQt5.QtCore as QtCore
from App_ui import Ui_MainWindow
from dialog_ui import Ui_Dialog

TESTING = True # Controls whether to use precomputed features

from ClassifierRunner import ClassifierRunner
from DataPoint import DataPoint
from Storage import Storage

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
    self.ui.showSegmentationsCheckbox.stateChanged.connect(self.ui.videoWidget.videoOverlay.setDrawSegmentations)
    self.ui.fileTableWidget.cellClicked.connect(self.videoInListClicked)
    self.ui.labelTableWidget.cellClicked.connect(self.labelInListClicked)
    self.processingProgressSignal.connect(self.processingProgressUpdate)
    self.processingCompleteSignal.connect(self.processingComplete)
    self.setWindowIcon(QIcon('icons/bosch.ico'))
    self.ui.actionInfo.triggered.connect(self.showInfoDialog)
    self.ui.actionProcessVideos.triggered.connect(self.initiateProcessing)

    # TODO what if user tries to process same video twice?
    self.dataPoints = dict()

    self.classifier = ClassifierRunner(TESTING)

    # just a thin wrapper around a storage device
    self.storage = Storage()

    # If we are in TESTING mode just load videos from the testing folder
    if TESTING:
      self.loadVideosFromFolder('testing/videos')
    else:
      self.ui.processMultipleFilesAction.triggered.connect(self.openFolderNameDialog)

    self.dialog = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(self.dialog)

  def showInfoDialog(self):
    self.dialog.show()

  def labelInListClicked(self, row, column):
    frameIndex = self.ui.labelTableWidget.currentItem().data(Qt.UserRole)
    self.ui.videoWidget.seekToTime(frameIndex)

  def setLabelList(self, dataPoint):
    self.ui.labelTableWidget.setRowCount(0)
    for label, labelTime in dataPoint.predictedLabels:
      rowIndex = self.ui.labelTableWidget.rowCount()
      self.ui.labelTableWidget.insertRow(rowIndex)
      item = QTableWidgetItem(label)
      item.setData(Qt.UserRole, labelTime)
      self.ui.labelTableWidget.setItem(rowIndex, 0, item)

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
    self.dataPoints[dataPoint.videoPath] = dataPoint

    #dataPoint.compareLabels()

    currentItem = self.ui.fileTableWidget.currentItem()
    if currentItem is not None:
      currentVideoPath = currentItem.data(Qt.UserRole)
      if currentVideoPath == dataPoint.videoPath:
        self.setCurrentVideo(dataPoint, play=False)

if __name__ == '__main__':
  import sys
  app = QApplication(sys.argv)
  window = MainWindow()
  window.show()
  sys.exit(app.exec_())
