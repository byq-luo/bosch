#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Sources Used:
ZetCode PyQt5 tutorial http://zetcode.com/gui/pyqt5/
"""

import sys
from PyQt5.QtWidgets import QFrame, QMainWindow, QTextEdit, QAction, QApplication, QFileDialog, QWidget, QLabel, \
    QDockWidget, QListWidget, QCheckBox
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QRectF
import PyQt5.QtCore as Qt
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt
#import PyQt5.QtGui
import PIL.Image, PIL.ImageTk
import time
import cv2
import math


class Home(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Home created")
        self.filename = None;
        self.time = int(round(time.time() * 1000))
        self.startUI()
        self.resize(1000, 600)
        self.vid = None
        self.paused = False
        self.currentFrame = 0

    def startUI(self):

        '''initializes the text showing the file path'''
        self.filetext = QLabel(self)
        self.filetext.setAlignment(Qt.AlignTop)
        self.filetext.setGeometry(10,10,30,80)
        self.filetext.setText("File: "+ str(self.filename))
        self.setCentralWidget(self.filetext)

        '''initializes the video screen'''
        self.videoscreen = QLabel(self)
        self.videoscreen.setGeometry(300, 0, 400, 500)
        self.setCentralWidget(self.videoscreen)
        img = QImage('icons/icon2.ico')
        i = img.scaled(720, 480, Qt.KeepAspectRatio)
        pmap = QPixmap(i)
        self.videoscreen.setPixmap(pmap)
        #pmap = QPixmap('icon2.ico')
        #p = pmap.scaled(640, 480, Qt.Qt.KeepAspectRatio)
        #self.videoscreen.setPixmap(p)

        self.labelList = QListWidget()
        self.labelList.setGeometry(700, 0, 300, 600)
        self.fileList = QListWidget()
        self.fileList.setGeometry(0, 0, 300, 600)
        self.boxCheckBox = QCheckBox("Display Box Over Target Object", self)
        self.boxCheckBox.setGeometry(400, 500, 400, 50)
        self.labelCheckBox = QCheckBox("Display Current Label", self)
        self.labelCheckBox.setGeometry(400, 550, 400, 50)


        self.rightDock = QDockWidget("Labels ", self)
        self.rightDock.setWidget(self.labelList)
        self.leftDock = QDockWidget("Files", self)
        self.leftDock.setWidget(self.fileList)
        self.bottomDock = QDockWidget("Box Display Option", self)
        self.bottomDock.setWidget(self.boxCheckBox)
        self.bottomDock2 = QDockWidget("Label Display Option", self)
        self.bottomDock2.setWidget(self.labelCheckBox)



        self.addDockWidget(Qt.RightDockWidgetArea, self.rightDock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.leftDock)
        self.setDockNestingEnabled(True)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottomDock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottomDock2)

        self.labelList.addItem("end,1.7207001116268856")
        self.labelList.addItem("evtEnd,67.567842448350611")
        self.labelList.addItem("rightTO=24,90.52518677490954")
        self.labelList.addItem("evtEnd,104.18015323449663")
        self.labelList.addItem("end,108,44646106956341")

        '''action to exit'''
        exitAct = QAction(QIcon('icons/exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.close)

        '''action to open one file'''
        fileAct = QAction(QIcon('icons/file.png'), 'Process File', self)
        #exitAct.setShortcut('Ctrl+Q')
        fileAct.setStatusTip('Process one video file')
        fileAct.triggered.connect(self.openFileNameDialog)

        '''action to open a whole folder'''
        folderAct = QAction(QIcon('icons/folder.png'), 'Process Folder', self)
        #folderAct.setShortcut('Ctrl+Q')
        folderAct.setStatusTip('Process every video file in a folder')
        #folderAct.triggered.connect(self.close)

        '''action to play video'''
        playAct = QAction(QIcon(''), 'Play', self)
        #playAct.setShortcut('Clt+Q')
        playAct.setStatusTip('Play the video')
        playAct.triggered.connect(self.makeVideo)

        '''action to pause video'''
        pauseAct = QAction(QIcon(''), 'Pause', self)
        pauseAct.setStatusTip('Pause the video')
        pauseAct.triggered.connect(self.pauseVideo)

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
        toolbar = self.addToolBar('PAUSE')
        toolbar.addAction(pauseAct)


        self.setGeometry(300, 300, 800, 540)   # sizes the window (x location, y location, width, height)
        self.setWindowTitle('Label Classifier')
        self.show()


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  ";AVI Files (*.avi)", options=options)
        if fileName:
            self.filename = fileName
            self.fileList.addItem(fileName)

    def makeVideo(self):
        if self.vid is None:
            if self.filename is not None:
                self.vid = Video(self.filename)
                self.vid.set_frame_number(self.currentFrame)
                self.currentFrame = 0
                self.delay = int(1000 / self.vid.get_fps())
                #self.update()
                #self.thread = QThread()
                #self.thread.start()



                self.worker = Thread(self.vid, self.delay)
                self.worker.changePixmap.connect(self.updateVidImage)

                #videoThread = QThread()

                #self.worker.moveToThread(self.thread)
                #worker = self.Thread()
                #orker.start()
                self.worker.start()

    @pyqtSlot(QImage)
    def updateVidImage(self, img):
        if not self.paused:
            pmap = QPixmap.fromImage(img)
            self.videoscreen.setPixmap(pmap)
            #self.show()

    def pauseVideo(self):
        self.paused = True
        self.worker.pause()
        self.currentFrame = self.vid.get_frame_number()
        self.vid = None
        self.worker.exit()


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, video, delay):
        super().__init__()
        print('Thread created')
        self.vid = video
        print('video saved')
        self.delay = delay
        self.paused2 = False

        #self.run()

    def run(self):
        if not self.paused2:
            self.time = int(round(time.time() * 1000))
            uptime = self.time + self.delay  # the time that the next frame should be pulled
            if self.vid is not None:
                ret, frame = self.vid.get_frame()
                if frame is None:
                    self.vid = None
                    """video has played all the way through"""
                if ret:
                    h, w, ch = frame.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480)
                    self.changePixmap.emit(p)
                    self.time = int(round(time.time() * 1000))
                    time.sleep((uptime - self.time) / 1000)  # call the function again after the difference in time has passed
                    self.run()

    def pause(self):
        self.paused2 = True




class Video:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

        # https://www.pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
        self.num_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        frameAvailable = False
        if self.vid.isOpened():
            frameAvailable, frame = self.vid.read()
            if frameAvailable:
                return (frameAvailable, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (frameAvailable, None)
        else:
            return(frameAvailable, None)

    def get_fps(self):
        return self.fps

    def get_total_num_frames(self):
        return self.num_frames

    def get_frame_number(self):
        return self.vid.get(cv2.CAP_PROP_POS_FRAMES)

    def set_frame_number(self, frame):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Home()
    sys.exit(app.exec_())