from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint, QTimer
from PyQt5.QtGui import QPainter, QImage

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
  
  def pause(self, toggled=False):
    self.isPlaying = False
    self.timer.stop()
    self.update()
  
  def play(self, toggled=False):
    if self.video is None:
      return
    self.isPlaying = True
    self.timer.stop()
    self.timer.start(self.video.get_fps())
    self.update()
  
  def seekToPercent(self, percent):
    if self.video is None:
      return
    totalNumFrames = self.video.get_total_num_frames()
    self.video.set_frame_number(int(percent / 100 * totalNumFrames))
    self.didSeek = True
    self.update()
  
  def setSlider(self, slider):
    self.slider = slider
  
  def setVideoPath(self, videoPath: str):
    self.video = Video(videoPath)
    # TODO
    #self.videoOverlay = VideoOverlay(videoPath)
    self.play()
  
  def __drawImage(self, frame, qp):
    vidHeight, vidWidth, vidChannels = frame.shape
    bytesPerLine = vidChannels * vidWidth
 
    widgetWidth = self.width()
    widgetHeight = self.height()
    if widgetWidth <= widgetHeight:
      scaledHeight = widgetWidth
      scaledWidth = widgetWidth
    if widgetWidth > widgetHeight:
      scaledWidth = widgetHeight
      scaledHeight = widgetHeight
 
    # TODO respect aspect ratio
 
    image = QImage(frame.data, vidWidth, vidHeight, bytesPerLine, QImage.Format_RGB888)
    image = image.scaled(scaledWidth, scaledHeight)
    putImageHere = QPoint(widgetWidth // 2 - scaledWidth // 2, widgetHeight // 2 - scaledHeight // 2)
    qp.drawImage(putImageHere, image)
  
  def paintEvent(self, e):
    qp = QPainter()
    qp.begin(self)
  
    if (self.isPlaying or self.didSeek) and self.video is not None:
      self.didSeek = False
      currentPercent = int(100 * self.video.get_frame_number() / self.video.get_total_num_frames())
      self.slider.setValue(currentPercent)
  
      frameAvailable, frame = self.video.get_frame()

      # TODO
      # frame = self.videoOverlay.processFrame(frame, frameNumber)

      # video has played all the way through
      if frame is None:
        self.pause()
        self.video.set_frame_number(0)
      elif frameAvailable:
        self.previousFrame = frame
        self.__drawImage(frame, qp)
    elif self.previousFrame is not None:
      self.__drawImage(self.previousFrame, qp)
    
    qp.end()

