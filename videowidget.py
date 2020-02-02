from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint, QTimer
from PyQt5.QtGui import QPainter, QImage

from Video import Video
import time

class VideoWidget(QWidget):
  def __init__(self, centralWidget):
    super().__init__()
    self.initUI()
  
  def initUI(self):
    self.video = None
    self.isPlaying = False

    # QTimer is basically what we were doing with time.sleep but more accurate
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update)

    # so that the image does not disappear when pause is hit
    self.previousFrame = None
  
  def pause(self, toggled: bool):
    self.isPlaying = False
    self.timer.stop()
  
  def play(self, toggled: bool):
    if self.video is None:
      return
    self.isPlaying = True
    self.timer.start(self.video.get_fps())
  
  def seekToPercent(self, percent):
    print('seeking')
    if self.video is None:
      return
    totalNumFrames = self.video.get_total_num_frames()
    self.video.set_frame_number(int(percent / 100 * totalNumFrames))
  
  def setSlider(self, slider):
    self.slider = slider
  
  def setVideoPath(self, videoPath: str):
    self.video = Video(videoPath)
    self.isPlaying = True
  
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
    videoPadding = 100
    scaledWidth = max(scaledWidth - 2 * videoPadding, 200)
    scaledHeight = max(scaledHeight - 2 * videoPadding, 200)
 
    # TODO respect aspect ratio
 
    image = QImage(frame.data, vidWidth, vidHeight, bytesPerLine, QImage.Format_RGB888)
    image = image.scaled(scaledWidth, scaledHeight)
    putImageHere = QPoint(widgetWidth // 2 - scaledWidth // 2, widgetHeight // 2 - scaledHeight // 2)
    qp.drawImage(putImageHere, image)
  
  def paintEvent(self, e):
    qp = QPainter()
    qp.begin(self)
  
    if self.isPlaying and self.video is not None:
      currentPercent = int(100 * self.video.get_frame_number() / self.video.get_total_num_frames())
      self.slider.setValue(currentPercent)
  
      frameAvailable, frame = self.video.get_frame()
      # video has played all the way through
      if frame is None:
        self.isPlaying = False
        self.video.set_frame_number(0)
      elif frameAvailable:
        self.previousFrame = frame
        self.__drawImage(frame, qp)
    elif self.previousFrame is not None:
      self.__drawImage(self.previousFrame, qp)
    
    qp.end()

