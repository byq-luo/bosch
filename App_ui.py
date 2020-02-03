# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'App.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        '''Make the main window'''
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1572, 675)

        '''Set up the central widget'''
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        '''Create the widget layouts'''
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")

        '''Add the file list widget'''
        self.verticalLayout_2.addWidget(self.label_2)
        self.fileListWidget = QtWidgets.QListWidget(self.centralwidget)
        self.fileListWidget.setObjectName("fileListWidget")
        self.verticalLayout_2.addWidget(self.fileListWidget)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        '''Add a spacer item'''
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)

        '''Add the video widget and set its size'''
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.videoWidget = VideoWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.videoWidget.sizePolicy().hasHeightForWidth())
        self.videoWidget.setSizePolicy(sizePolicy)
        self.videoWidget.setMinimumSize(QtCore.QSize(10, 0))
        self.videoWidget.setObjectName("videoWidget")
        self.verticalLayout_6.addWidget(self.videoWidget)

        '''Add the play button'''
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.playButton = QtWidgets.QPushButton(self.centralwidget)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_2.addWidget(self.playButton)

        '''Add a spacer item'''
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)

        '''Add a horizontal slider'''
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setInvertedControls(False)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_2.addWidget(self.horizontalSlider)

        '''Add a spacer item'''
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)

        '''Add the pause button'''
        self.pauseButton = QtWidgets.QPushButton(self.centralwidget)
        self.pauseButton.setObjectName("pauseButton")
        self.horizontalLayout_2.addWidget(self.pauseButton)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_6)

        '''Add a spacer item'''
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)

        '''Add label list widget'''
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.labelListWidget = QtWidgets.QListWidget(self.centralwidget)
        self.labelListWidget.setObjectName("labelListWidget")
        self.verticalLayout.addWidget(self.labelListWidget)

        '''Add showLabels checkbox'''
        self.showLabelsCheckbox = QtWidgets.QCheckBox(self.centralwidget)
        self.showLabelsCheckbox.setObjectName("showLabelsCheckbox")
        self.verticalLayout.addWidget(self.showLabelsCheckbox)

        '''Add showBoundingBox checkbox'''
        self.boundingBoxCheckbox = QtWidgets.QCheckBox(self.centralwidget)
        self.boundingBoxCheckbox.setObjectName("boundingBoxCheckbox")
        self.verticalLayout.addWidget(self.boundingBoxCheckbox)
        self.horizontalLayout.addLayout(self.verticalLayout)

        '''Set central widget'''
        MainWindow.setCentralWidget(self.centralwidget)

        '''Add status bar'''
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        '''Add tool bar'''
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        '''Add file select option to toolbar'''
        self.processOneFileAction = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.processOneFileAction.setIcon(icon)
        self.processOneFileAction.setText("")
        self.processOneFileAction.setObjectName("processOneFileAction")

        '''Add folder select option to toolbar'''
        self.processMultipleFilesAction = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.processMultipleFilesAction.setIcon(icon1)
        self.processMultipleFilesAction.setText("")
        self.processMultipleFilesAction.setObjectName("processMultipleFilesAction")

        '''Add action to play button'''
        self.playAction = QtWidgets.QAction(MainWindow)
        self.playAction.setObjectName("playAction")

        '''Add action to pause button'''
        self.pauseAction = QtWidgets.QAction(MainWindow)
        self.pauseAction.setObjectName("pauseAction")

        '''Add action to file select'''
        self.toolBar.addAction(self.processOneFileAction)

        '''Add tool bar separator'''
        self.toolBar.addSeparator()

        '''Add action to folder select'''
        self.toolBar.addAction(self.processMultipleFilesAction)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Videos"))
        self.playButton.setText(_translate("MainWindow", "Play"))
        self.pauseButton.setText(_translate("MainWindow", "Pause"))
        self.label.setText(_translate("MainWindow", "Labels"))
        self.showLabelsCheckbox.setText(_translate("MainWindow", "Show Box Over Target Object"))
        self.boundingBoxCheckbox.setText(_translate("MainWindow", "Show Current Label"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.processOneFileAction.setToolTip(_translate("MainWindow", "Process One Video File"))
        self.processMultipleFilesAction.setToolTip(_translate("MainWindow", "Recursively Process Videos From Folder"))
        self.playAction.setText(_translate("MainWindow", "Play"))
        self.pauseAction.setText(_translate("MainWindow", "Pause"))
from videowidget import VideoWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
