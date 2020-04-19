# https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies
# http://blog.rcnelson.com/building-a-matplotlib-gui-with-qt-designer-part-2/
from PyQt5.QtWidgets import QDialog, QLabel, QApplication, QPushButton, QVBoxLayout
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
from DataPoint import DataPoint

import torch

from infodialog_ui import Ui_InfoDialog


def getBold(s):
  return '<span style=" font-weight:600;">' + s + '</span>'


def getTxt(k, v):
  return '<html><head/><body><p>' + getBold(k) + v + '</p></body></html>'


class InfoDialog(QDialog):
  def __init__(self, dataPoints: dict):
    super(InfoDialog, self).__init__()

    # Set up the user interface from Designer.
    self.ui = Ui_InfoDialog()
    self.ui.setupUi(self)
    self.canvas = None
    self.labelCounts = {}

    # self.estHoursSaved = ''
    self.hoursOfVidProcessed = ''
    self.processingHoursRemaining = ''
    self.averageVidLength = ''
    self.deviceName = "Cuda not available!"
    if torch.cuda.is_available():
      self.deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
    self.updateState(dataPoints)

  def _updateLabels(self, dataPoints : list):
    videoTimes = [dp.videoLength for dp in dataPoints]
    totalTime = sum(videoTimes) / 3600
    avgTime = int(totalTime / (len(videoTimes)+1) * 60)

    videoTimesProcessed = [dp.videoLength for dp in dataPoints if dp.hasBeenProcessed]
    hoursProcessed = int(sum(videoTimesProcessed) / 3600)
    hoursRemaining = int(totalTime - hoursProcessed)

    # self.estHoursSaved = str('idk')
    self.hoursOfVidProcessed = str(hoursProcessed)
    self.processingHoursRemaining = str(hoursRemaining)
    self.averageVidLength = str(avgTime) + ' minutes'

    # self.ui.label_1.setText(getTxt('Estimated hours saved: ', self.estHoursSaved))
    self.ui.label_2.setText(getTxt('Hours of video processed: ', self.hoursOfVidProcessed))
    self.ui.label_3.setText(getTxt('Processing hours remaining: ', self.processingHoursRemaining))
    self.ui.label_4.setText(getTxt('Average video length: ', self.averageVidLength))
    self.ui.label_5.setText(getTxt('GPU device name: ', self.deviceName))

  def _updateHist(self, dataPoints : list):
    self.labelCounts = dict()
    for dp in dataPoints:
      if dp.hasBeenProcessed:
        for label, frameNum in dp.predictedLabels:
          self.labelCounts[label] = self.labelCounts.get(label, 0) + 1
    self.setFig(self.genPlot())

  def updateState(self, dataPoints : dict):
    dataPoints = dataPoints.values()
    self._updateLabels(dataPoints)
    self._updateHist(dataPoints)

  def setFig(self, fig):
    if self.canvas is not None:
      self.ui.plotLayout.removeWidget(self.canvas)
      self.canvas.close()
    self.canvas = FigureCanvas(fig)
    self.canvas.setStyleSheet("background-color:transparent;")
    self.ui.plotLayout.addWidget(self.canvas)
    self.canvas.draw()

  def genPlot(self):
    fig = Figure()
    fig.patch.set_facecolor("None")
    ax = fig.add_subplot(111)
    heights = self.labelCounts.values()
    bars = self.labelCounts.keys()
    y_pos = range(len(bars))
    ax.bar(y_pos, heights)
    ax.set_xticks(np.arange(len(bars)))
    ax.set_xticklabels(bars, rotation=45)
    ax.set_title('Predicted label frequencies')
    # Tweak spacing to prevent clipping of tick-labels
    fig.subplots_adjust(bottom=0.25)

    return fig
