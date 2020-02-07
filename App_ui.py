# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'App.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QListWidgetItem

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1032, 576)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.fileTableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.fileTableWidget.setMaximumSize(QtCore.QSize(375, 16777215))
        self.fileTableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.fileTableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fileTableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fileTableWidget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.fileTableWidget.setShowGrid(False)
        self.fileTableWidget.setCornerButtonEnabled(False)
        self.fileTableWidget.setObjectName("fileTableWidget")
        self.fileTableWidget.setColumnCount(2)
        self.fileTableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.fileTableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.fileTableWidget.setHorizontalHeaderItem(1, item)
        self.fileTableWidget.horizontalHeader().setDefaultSectionSize(60)
        self.fileTableWidget.horizontalHeader().setHighlightSections(False)
        self.fileTableWidget.horizontalHeader().setStretchLastSection(False)
        self.fileTableWidget.verticalHeader().setVisible(False)
        self.verticalLayout_2.addWidget(self.fileTableWidget)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
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
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.currentVideoTime = QtWidgets.QLabel(self.centralwidget)
        self.currentVideoTime.setObjectName("currentVideoTime")
        self.horizontalLayout_3.addWidget(self.currentVideoTime)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setInvertedControls(False)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_3.addWidget(self.horizontalSlider)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.fullVideoTime = QtWidgets.QLabel(self.centralwidget)
        self.fullVideoTime.setObjectName("fullVideoTime")
        self.horizontalLayout_3.addWidget(self.fullVideoTime)
        self.verticalLayout_6.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.playButton = QtWidgets.QPushButton(self.centralwidget)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_2.addWidget(self.playButton)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.pauseButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pauseButton.sizePolicy().hasHeightForWidth())
        self.pauseButton.setSizePolicy(sizePolicy)
        self.pauseButton.setObjectName("pauseButton")
        self.horizontalLayout_2.addWidget(self.pauseButton)
        spacerItem5 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_6)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelTableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.labelTableWidget.setMaximumSize(QtCore.QSize(250, 16777215))
        self.labelTableWidget.setShowGrid(False)
        self.labelTableWidget.setObjectName("labelTableWidget")
        self.labelTableWidget.setColumnCount(1)
        self.labelTableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.labelTableWidget.setHorizontalHeaderItem(0, item)
        self.labelTableWidget.horizontalHeader().setHighlightSections(False)
        self.labelTableWidget.horizontalHeader().setStretchLastSection(True)
        self.labelTableWidget.verticalHeader().setVisible(False)
        self.verticalLayout.addWidget(self.labelTableWidget)
        self.showLabelsCheckbox = QtWidgets.QCheckBox(self.centralwidget)
        self.showLabelsCheckbox.setObjectName("showLabelsCheckbox")
        self.verticalLayout.addWidget(self.showLabelsCheckbox)
        self.boundingBoxCheckbox = QtWidgets.QCheckBox(self.centralwidget)
        self.boundingBoxCheckbox.setObjectName("boundingBoxCheckbox")
        self.verticalLayout.addWidget(self.boundingBoxCheckbox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.processOneFileAction = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.processOneFileAction.setIcon(icon)
        self.processOneFileAction.setText("")
        self.processOneFileAction.setObjectName("processOneFileAction")
        self.processMultipleFilesAction = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.processMultipleFilesAction.setIcon(icon1)
        self.processMultipleFilesAction.setText("")
        self.processMultipleFilesAction.setObjectName("processMultipleFilesAction")
        self.actionProcessVideos = QtWidgets.QAction(MainWindow)
        self.actionProcessVideos.setObjectName("actionProcessVideos")
        self.toolBar.addAction(self.processOneFileAction)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.processMultipleFilesAction)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionProcessVideos)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Label Generator"))
        self.fileTableWidget.setSortingEnabled(True)
        item = self.fileTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Score"))
        item = self.fileTableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Filename"))
        self.currentVideoTime.setText(_translate("MainWindow", "00:00"))
        self.fullVideoTime.setText(_translate("MainWindow", "00:00"))
        self.playButton.setText(_translate("MainWindow", "Play"))
        self.pauseButton.setText(_translate("MainWindow", "Pause"))
        item = self.labelTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Predictions"))
        self.showLabelsCheckbox.setText(_translate("MainWindow", "Show Current Label"))
        self.boundingBoxCheckbox.setText(_translate("MainWindow", "Show Bounding Box"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.processOneFileAction.setToolTip(_translate("MainWindow", "Process One Video File"))
        self.processMultipleFilesAction.setToolTip(_translate("MainWindow", "Recursively Process Videos From Folder"))
        self.actionProcessVideos.setText(_translate("MainWindow", "Process Videos"))
from videowidget import VideoWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
