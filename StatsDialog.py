# https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies
# http://blog.rcnelson.com/building-a-matplotlib-gui-with-qt-designer-part-2/
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os

from statsdialog_ui import Ui_StatsDialog

labels = []
def genFakeData():
    for ls in os.walk('labels'):
        for g in ls:
            for f in g:
                name, ext = os.path.splitext(f)
                if ext == '.txt':
                    with open('labels/'+f) as file:
                        for line in file.readlines():
                            label, labelTime = line.split(',')
                            label = label.split('=')[0]
                            labels.append(label)

class StatsDialog(QDialog):
    def __init__(self):
        super(StatsDialog, self).__init__()

        # Set up the user interface from Designer.
        self.ui = Ui_StatsDialog()
        self.ui.setupUi(self)

        genFakeData()

        self.canvas = None
        self.fig = Figure()
        self.fig.patch.set_facecolor("None")
        ax = self.fig.add_subplot(111)
        ax.set_title('Predicted label frequencies')
        ax.hist(labels, rwidth=.8)
        self.set_fig(self.fig)

    def set_fig(self, fig):
        if self.canvas is not None:
            self.ui.plotLayout.removeWidget(self.canvas)
            self.canvas.close()
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.ui.plotLayout.addWidget(self.canvas)
        self.canvas.draw()