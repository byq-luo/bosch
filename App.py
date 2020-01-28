#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Sources Used:
ZetCode PyQt5 tutorial http://zetcode.com/gui/pyqt5/
"""

import sys
from PyQt5.QtWidgets import QFrame, QMainWindow, QTextEdit, QAction, QApplication, QFileDialog, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QRectF
import PyQt5.QtCore as Qt
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

        '''initializes the text showing the file path'''
        self.filetext = QLabel(self)
        self.filetext.setAlignment(Qt.Qt.AlignTop)
        self.filetext.setGeometry(10,10,30,80)
        self.filetext.setText("File: "+ str(self.filename))
        self.setCentralWidget(self.filetext)

        '''initializes the video screen'''
        self.videoscreen = QLabel(self)
        self.setCentralWidget(self.videoscreen)
        img = QImage('icon2.ico')
        i = img.scaled(720, 480, Qt.Qt.KeepAspectRatio)
        pmap = QPixmap(i)
        self.videoscreen.setPixmap(pmap)
        #pmap = QPixmap('icon2.ico')
        #p = pmap.scaled(640, 480, Qt.Qt.KeepAspectRatio)
        #self.videoscreen.setPixmap(p)

        '''action to exit'''
        exitAct = QAction(QIcon('exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.close)

        '''action to open one file'''
        fileAct = QAction(QIcon('file.png'), 'Process File', self)
        #exitAct.setShortcut('Ctrl+Q')
        fileAct.setStatusTip('Process one video file')
        fileAct.triggered.connect(self.openFileNameDialog)

        '''action to open a whole folder'''
        folderAct = QAction(QIcon('folder.png'), 'Process Folder', self)
        #folderAct.setShortcut('Ctrl+Q')
        folderAct.setStatusTip('Process every video file in a folder')
        #folderAct.triggered.connect(self.close)

        '''action to play video'''
        playAct = QAction(QIcon(''), 'Play', self)
        #playAct.setShortcut('Clt+Q')
        playAct.setStatusTip('Play the video')
        playAct.triggered.connect(self.makeVideo)

        self.statusBar()

        menubar = self.menuBar()

        '''Add actions to menubar'''
        actionMenu = menubar.addMenu('&Action')
        actionMenu.addAction(exitAct)
        actionMenu.addAction(fileAct)
        actionMenu.addAction(folderAct)
        #actionMenu.addAction(playAct)

        '''Add actions to toolbar'''
        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAct)
        toolbar = self.addToolBar('OPEN FILE')
        toolbar.addAction(fileAct)
        toolbar = self.addToolBar('OPEN FOLDER')
        toolbar.addAction(folderAct)
        toolbar = self.addToolBar('PLAY')
        toolbar.addAction(playAct)

        self.setGeometry(300, 300, 800, 540)   # sizes the window (x location, y location, width, height)
        self.setWindowTitle('Label Classifier')
        self.show()

    class Thread(QThread):
        def __init__(self, video, delay):
            super().__init__()
            self.vid = video
            self.delay = delay
        changePixmap = pyqtSignal(QImage)

        def run(self):
            self.time = int(round(time.time() * 1000))
            uptime = self.time + self.delay  # the time that the next frame should be pulled
            if self.vid is not None:
                ret, frame = self.vid.get_frame()
                if frame is None:
                    """video has played all the way through"""

                    #return False
                if ret:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480)
                    #pmap = QPixmap(convertToQtFormat)
                    # self.videoscreen.setPixmap(pmap)
                    # self.resize(convertToQtFormat.width(), convertToQtFormat.height())
                    # self.show()
                    self.changePixmap.emit(p)
                    self.time = int(round(time.time() * 1000))
                    time.sleep((uptime - self.time) / 1000)  # call the function again after the difference in time has passed
                    self.run()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  ";AVI Files (*.avi)", options=options)
        if fileName:
            self.filename = fileName
            #self.filetext.setText('File: ' + str(self.filename))

    def makeVideo(self):
        if self.filename is not None:
            self.vid = gui.Video(self.filename)
            self.delay = int(1000 / self.vid.get_fps())
            #self.update()
            worker = self.Thread(self.vid, self.delay)
            worker.changePixmap.connect(self.updateVidImage)
            worker.start()

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
                p = convertToQtFormat.scaled(720, 480, Qt.Qt.KeepAspectRatio)
                pmap = QPixmap(p)
                self.videoscreen.setPixmap(pmap)
                #self.resize(convertToQtFormat.width(), convertToQtFormat.height())
                #self.videoscreen.show()

        self.time = int(round(time.time() * 1000))
        time.sleep((uptime - self.time)/1000)  # call the function again after the difference in time has passed
        self.update()

    @pyqtSlot(QImage)
    def updateVidImage(self, img):
        pmap = QPixmap.fromImage(img)
        self.videoscreen.setPixmap(pmap)
        #self.show()

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