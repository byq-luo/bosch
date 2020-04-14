# https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies
# http://blog.rcnelson.com/building-a-matplotlib-gui-with-qt-designer-part-2/
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
from DataPoint import DataPoint

import torch
deviceName = "Cuda not available!"
if torch.cuda.is_available():
  deviceName = torch.cuda.get_device_name(torch.cuda.current_device())

from statsdialog_ui import Ui_StatsDialog


class StatsDialog(QDialog):
  def __init__(self):
    super(StatsDialog, self).__init__()

    # Set up the user interface from Designer.
    self.ui = Ui_StatsDialog()
    self.ui.setupUi(self)
    self.canvas = None
    self.labelCounts = {}

    self.set_fig(self.genPlot())


  def updatePlot(self, dp : DataPoint):
    for label, frameNum in dp.predictedLabels:
      self.labelCounts[label] = self.labelCounts.get(label, 0) + 1
    self.set_fig(self.genPlot())

  def set_fig(self, fig):
    if self.canvas is not None:
      self.ui.plotLayout.removeWidget(self.canvas)
      self.canvas.close()
    self.canvas = FigureCanvas(fig)
    self.canvas.setStyleSheet("background-color:transparent;")
    self.ui.plotLayout.addWidget(self.canvas)
    self.ui.label_8.setText(deviceName)
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
