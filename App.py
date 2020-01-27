#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Sources Used:
ZetCode PyQt5 tutorial http://zetcode.com/gui/pyqt5/
"""

import sys
from PyQt5.QtWidgets import QFrame, QMainWindow, QTextEdit, QAction, QApplication, QFileDialog, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
#import PyQt5.QtGui
import gui
import PIL.Image, PIL.ImageTk
import time
import cv2
import math


class Home(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filename = None;
        self.time = int(round(time.time() * 1000))
        self.startUI()

    def startUI(self):

        self.videoscreen = QLabel(self)
        self.setCentralWidget(self.videoscreen)
        pmap = QPixmap('icon2.ico')
        self.videoscreen.setPixmap(pmap)


        exitAct = QAction(QIcon('exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.close)

        fileAct = QAction(QIcon('file.png'), 'Process File', self)
        #exitAct.setShortcut('Ctrl+Q')
        fileAct.setStatusTip('Process one video file')
        fileAct.triggered.connect(self.openFileNameDialog)

        folderAct = QAction(QIcon('folder.png'), 'Process Folder', self)
        #folderAct.setShortcut('Ctrl+Q')
        folderAct.setStatusTip('Process every video file in a folder')
        #folderAct.triggered.connect(self.close)

        playAct = QAction(QIcon(''), 'Play', self)
        #playAct.setShortcut('Clt+Q')
        playAct.setStatusTip('Play the video')
        playAct.triggered.connect(self.makeVideo)

        self.statusBar()

        menubar = self.menuBar()
        actionMenu = menubar.addMenu('&Action')
        actionMenu.addAction(exitAct)
        actionMenu.addAction(fileAct)
        actionMenu.addAction(folderAct)
        #actionMenu.addAction(playAct)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAct)
        toolbar = self.addToolBar('OPEN FILE')
        toolbar.addAction(fileAct)
        toolbar = self.addToolBar('OPEN FOLDER')
        toolbar.addAction(folderAct)
        toolbar = self.addToolBar('PLAY')
        toolbar.addAction(playAct)

        self.setGeometry(300, 300, 640, 480)   # sizes the window (x location, y location, width, height)
        self.setWindowTitle('Label Classifier')
        self.show()

    '''class Thread(QThread):
        changePixmap = pyqtSignal(QImage)

        def run(self):
            if self.vid is not None:
                ret, frame = self.vid.get_frame()
                if frame is None:
                    """video has played all the way through"""
                    return False
                if ret:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    # p = convertToQtFormat.scaled(640, 480, QImage.KeepAspectRatio)
                    pmap = QPixmap(convertToQtFormat)
                    self.videoscreen.setPixmap(pmap)
                    # self.resize(convertToQtFormat.width(), convertToQtFormat.height())
                    self.show()
                return True
            return False'''

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  ";AVI Files (*.avi)", options=options)
        if fileName:
            self.filename = fileName

    def makeVideo(self):
        if self.filename is not None:
            self.vid = gui.Video(self.filename)
            self.delay = int(1000 / self.vid.get_fps())
            self.update()
            '''worker = Worker(self)
            worker.changePixmap.connect(self.updateVidImage)
            worker.start()'''

    def update(self):
        self.time = int(round(time.time() * 1000))
        uptime = self.time + self.delay  # the time that the next frame should be pulled
        if self.vid is not None:
            ret, frame = self.vid.get_frame()
            if frame is None:
                """video has played all the way through"""
                return

            #self.draw_bounding_box(uptime, frame)

            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                #p = convertToQtFormat.scaled(640, 480, QImage.KeepAspectRatio)
                pmap = QPixmap(convertToQtFormat)
                self.videoscreen.setPixmap(pmap)
                #self.resize(convertToQtFormat.width(), convertToQtFormat.height())
                #self.videoscreen.show()

        self.time = int(round(time.time() * 1000))
        time.sleep((uptime - self.time)/1000)  # call the function again after the difference in time has passed
        self.update()

    #@pyqtSignal()
    def updateVidImage(self, pmap):
        self.videoscreen.setPixmap(pmap)
        self.show()

'''
class Worker(QThread):
    changePixmap = pyqtSignal(QImage)
    def run(self):
        if self.vid is not None:
            ret, frame = self.vid.get_frame()
            if frame is None:
                """video has played all the way through"""
                return False
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                #p = convertToQtFormat.scaled(640, 480, QImage.KeepAspectRatio)
                pmap = QPixmap(convertToQtFormat)
                self.changePixmap.emit(pmap)
                #self.videoscreen.setPixmap(pmap)
                #self.resize(convertToQtFormat.width(), convertToQtFormat.height())
                #self.show()
        return True
'''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Home()
    sys.exit(app.exec_())