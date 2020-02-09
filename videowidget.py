from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint, QTimer, Qt
from PyQt5.QtGui import QPainter, QImage
from App import TESTING

if TESTING:
  from mock.Video import Video
else:
  from Video import Video

from VideoOverlay import VideoOverlay

import time

class VideoWidget(QWidget):
  def __init__(self, centralWidget):
    super().__init__()
    self.initUI()

  def initUI(self):
    self.video = None
    self.isPlaying = False
    self.didSeek = False

    # QTimer is basically what we were doing with time.sleep but more accurate
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update)

    # so that the image does not disappear when pause is hit
    self.previousFrame = None

    self.videoOverlay = VideoOverlay()

  def pause(self, toggled=False):
    self.isPlaying = False
    self.timer.stop()
    self.update()

  def play(self, toggled=False):
    if self.video is None:
      return
    self.isPlaying = True
    self.timer.stop()
    self.timer.start(self.video.getFps())

  def seekToFrame(self, frameIndex):
    self.video.setFrameNumber(frameIndex)
    self.didSeek = True
    self.update()

  def seekToPercent(self, percent):
    if self.video is None:
      return
    totalNumFrames = self.video.getTotalNumFrames()
    frameIndex = int(percent / 100 * totalNumFrames)
    self.seekToFrame(frameIndex)

  def setSlider(self, slider):
    self.slider = slider

  def setTimeLabels(self, currentTime, fullTime):
    self.currentTimeLabel = currentTime
    self.fullTimeLabel = fullTime

  def setVideo(self, dataPoint):
    self.dataPoint = dataPoint
    self.video = Video(dataPoint.videoPath)
    self.setFullTimeLabel()

  def setFullTimeLabel(self):
    totalSeconds = self.video.getVideoLength()
    minutes = int(totalSeconds / 60)
    seconds = int(totalSeconds % 60)
    timeString = str(minutes) + ":" + str(seconds)
    self.fullTimeLabel.setText(timeString)

  def updateTimeLabel(self):
    totalSeconds = self.video.getCurrentTime()
    minutes = int(totalSeconds / 60)
    seconds = int(totalSeconds % 60)
    timeString = '{}:{:02d}'.format(minutes, seconds)
    self.currentTimeLabel.setText(timeString)

  def updateSliderValue(self):
    currentPercent = int(100 * self.video.getFrameNumber() / self.video.getTotalNumFrames())
    self.slider.setValue(currentPercent)

  def _drawImage(self, frame, qp:QPainter):
    vidHeight, vidWidth, vidChannels = frame.shape
    bytesPerLine = vidChannels * vidWidth

    widgetWidth = self.width()
    widgetHeight = self.height()

    scaleHorizontal = widgetWidth / vidWidth
    scaleVertical = widgetHeight / vidHeight
    if scaleHorizontal < scaleVertical:
      scaledWidth = int(vidWidth*scaleHorizontal)
      scaledHeight = int(vidHeight*scaleHorizontal)
    else:
      scaledWidth = int(vidWidth*scaleVertical)
      scaledHeight = int(vidHeight*scaleVertical)

    image = QImage(frame.data, vidWidth, vidHeight, bytesPerLine, QImage.Format_RGB888)
    image = image.scaled(scaledWidth, scaledHeight, transformMode=Qt.SmoothTransformation)
    putImageHere = QPoint(widgetWidth // 2 - scaledWidth // 2, widgetHeight // 2 - scaledHeight // 2)
    qp.drawImage(putImageHere, image)

  def paintEvent(self, e):
    qp = QPainter()
    qp.begin(self)

    if (self.isPlaying or self.didSeek) and self.video is not None:
      self.updateTimeLabel()
      self.updateSliderValue()
      self.didSeek = False

      frameIndex = self.video.getFrameNumber()
      frameAvailable, frame = self.video.getFrame()

      frame = self.videoOverlay.processFrame(frame, frameIndex, self.dataPoint)

      # video has played all the way through
      if frame is None:
        self.pause()
        self.video.setFrameNumber(0)
      elif frameAvailable:
        self.previousFrame = frame
        self._drawImage(frame, qp)
    elif self.previousFrame is not None:
      self._drawImage(self.previousFrame, qp)

    qp.end()

