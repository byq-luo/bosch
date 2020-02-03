from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QListWidgetItem
from App_ui import Ui_MainWindow

# from ML.BehaviorClassifier import BehaviorClassifier

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
    self.ui.fileListWidget.currentTextChanged.connect(self.ui.videoWidget.setVideoPath)
    self.ui.videoWidget.setSlider(self.ui.horizontalSlider)

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
    folderName = QFileDialog.getExistingDirectory(self, caption="Select Directory",
                                                  options=options)
    if folderName:
      print(folderName)
      pass

import sys
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
