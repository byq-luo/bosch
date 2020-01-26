#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Sources Used:
ZetCode PyQt5 tutorial http://zetcode.com/gui/pyqt5/
"""

import sys
from PyQt5.QtWidgets import QFrame, QMainWindow, QTextEdit, QAction, QApplication
from PyQt5.QtGui import QIcon


class Home(QMainWindow):
    def __init__(self):
        super().__init__()

        self.startUI()

    def startUI(self):
        self.videoscreen = QFrame(self)
        self.videoscreen.setGeometry(150, 20, 100, 100)
        self.videoscreen.setStyleSheet("QWidget { background-color: black }")


        self.setCentralWidget(self.videoscreen)

        exitAct = QAction(QIcon('exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.close)

        fileAct = QAction(QIcon('file.png'), 'Process File', self)
        #exitAct.setShortcut('Ctrl+Q')
        fileAct.setStatusTip('Process one video file')
        #fileAct.triggered.connect(self.close)

        folderAct = QAction(QIcon('folder.png'), 'Process Folder', self)
        #folderAct.setShortcut('Ctrl+Q')
        folderAct.setStatusTip('Process every video file in a folder')
        #folderAct.triggered.connect(self.close)

        self.statusBar()

        menubar = self.menuBar()
        actionMenu = menubar.addMenu('&Action')
        actionMenu.addAction(exitAct)
        actionMenu.addAction(fileAct)
        actionMenu.addAction(folderAct)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAct)
        toolbar = self.addToolBar('OPEN FILE')
        toolbar.addAction(fileAct)
        toolbar = self.addToolBar('OPEN FOLDER')
        toolbar.addAction(folderAct)

        self.setGeometry(300, 300, 500, 500)   # sizes the window (x location, y location, width, height)
        self.setWindowTitle('Label Classifier')
        self.show()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Home()
    sys.exit(app.exec_())