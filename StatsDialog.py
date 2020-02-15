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

def get_color(num_curves_in_plot):
    colors = ['k','r','m','b','y','c','g']
    # we only have 7 colors but at most 16 targets
    # so add shape to the curve to make curves that end up with the same color distinct
    color = colors[num_curves_in_plot % len(colors)]
    if num_curves_in_plot // len(colors) > 1:
        color += '-o'
    elif num_curves_in_plot // len(colors) > 0:
        color += '-.'
    return color

def xx():
    # plot the multicomponent curves from the eds file
    figs = []

    # group curves by sample name
    df = pd.DataFrame(curves_to_plot, columns=['sample_name', 'target_name', 'curve'])
    df['sample_name'] = df['sample_name'].str.upper() # do not split a group based on case
    df_groups = df.groupby('sample_name')
    # make a figure for each sample name
    for _,group in df_groups:
        fig = Figure()
        ax = fig.add_subplot(111)
        target_names = []
        # place each curve for sample into figure
        num_curves_in_plot = 0
        for _,row in group.iterrows():
            x = np.arange(len(row['curve']))
            y = row['curve']
            ax.plot(x, y, get_color(num_curves_in_plot))
            target_names.append(row['target_name'])
            num_curves_in_plot += 1

        ax.legend(labels=target_names, loc='upper left')
        figs.append((row['sample_name'], fig))

    # sort plots based on sample name
    figs.sort(key=lambda x : x[0].lower())