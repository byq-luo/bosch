from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint, QTimer, Qt
from PyQt5.QtGui import QPainter, QImage
from VideoOverlay import VideoOverlay
from Video import Video
import CONFIG 

if CONFIG.IMMEDIATE_MODE:
  from LaneLineDetectorERFNet import LaneLineDetector
  from VehicleDetectorYolo import VehicleDetectorYolo
  from VehicleTrackerSORT import VehicleTracker
  import cv2
  import numpy as np

class VideoWidget(QWidget):
  def __init__(self, centralWidget):
    super().__init__()
    self.initUI()
    if CONFIG.IMMEDIATE_MODE:
      self.lane= LaneLineDetector()
      self.vehicleDetector = VehicleDetectorYolo()
      self.tracker = VehicleTracker()

  def initUI(self):
    self.video = None
    self.dataPoint = None

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

  def seekToTime(self, t):
    frameNum = int(t * self.video.getFps())
    numFrames = self.video.getTotalNumFrames()
    self.video.setFrameNumber(min(frameNum, numFrames-1))
    self.didSeek = True
    self.update()

  def seekToPercent(self, percent):
    if self.video is None:
      return
    totalNumFrames = self.video.getTotalNumFrames()
    frameIndex = int(percent / 1000 * totalNumFrames)
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
    timeString = '{:03d}'.format((int(totalSeconds)))
    self.fullTimeLabel.setText(timeString)

  def updateTimeLabel(self):
    totalSeconds = self.video.getCurrentTime()
    timeString = '{:03d}'.format(int(totalSeconds))
    self.currentTimeLabel.setText(timeString)

  def updateSliderValue(self):
    currentPercent = int(1000 * self.video.getFrameNumber() / self.video.getTotalNumFrames())
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
      isFrameAvail, frame = self.video.getFrame()
      currentTime = self.video.getCurrentTime()

      # video has played all the way through
      if frame is None:
        self.pause()
        self.video.setFrameNumber(0)
      elif isFrameAvail:
        if CONFIG.IMMEDIATE_MODE:
          # frame = cv2.blur(frame, (2,2))
          frame = self.renderImmediately(frame, frameIndex)
        else:
          frame = self.videoOverlay.processFrame(frame, frameIndex, self.dataPoint, currentTime)
        self.previousFrame = frame
        self._drawImage(frame, qp)
    elif self.previousFrame is not None:
      self._drawImage(self.previousFrame, qp)

    qp.end()

  # This function is only for development purposes
  def renderImmediately(self, frame, frameIndex):
      # rawboxes, boxscores = self.vehicleDetector.getFeatures(frame)
      # vehicles = self.tracker.getVehicles(frame, rawboxes, boxscores)
      # # Use VehicleTrackerDL to render fingerprint of vehicles
      # # frame = self.tracker.getVehicles(frame, rawboxes, boxscores)
      # self.boundingBoxColor = (0,255,0)
      # self.labelColor = (255,0,0)
      # self.boundingBoxThickness = 1
      # # Sending back min length box list works good
      # vehicleBoxes = [v.box for v in vehicles]
      # vehicleIDs = [v.id for v in vehicles]
      # if len(vehicleBoxes) < len(rawboxes):
      #   vehicleBoxes += [np.array([0, 0, 0, 0])] * (len(rawboxes)-len(vehicleBoxes))
      #   vehicleIDs += ['.']
      # if len(vehicleBoxes) > len(rawboxes):
      #   rawboxes += [np.array([0, 0, 0, 0])] * (len(vehicleBoxes)-len(rawboxes))
      # bboxes = []
      # for box, vbox, _id in zip(rawboxes, vehicleBoxes, vehicleIDs):
      #   bboxes.append((list(map(int, box)), list(map(int, vbox)), _id))
      # for (bx1,by1,bx2,by2),(x1,y1,x2,y2),_id in bboxes:
      #   x, y = x1,y1-7
      #   cv2.rectangle(frame,
      #     (x1,y1), (x2,y2),
      #     self.boundingBoxColor,
      #     self.boundingBoxThickness)
      #   cv2.rectangle(frame,
      #     (bx1,by1), (bx2,by2),
      #     (0,0,255),
      #     self.boundingBoxThickness)
      #   cv2.putText(frame,
      #               str(_id),
      #               (x,y),
      #               0, .3,
      #               self.labelColor)
      # cv2.putText(frame,
      #             str(len(rawboxes)),
      #             (30,30),
      #             0, 1,
      #             self.labelColor)

      laneColors = [(255,0,0),(255,255,0),(0,255,0),(0,0,255)]
      lines = self.lane.getLines(frame)
      for i,line in enumerate(lines):
        for (x1, y1, x2, y2) in line:
          cv2.line(frame, (x1, y1), (x2, y2), laneColors[i], 2)

      # Visualize probability maps
      # ls = .5*l  + .5*r
      # ls = np.array([ls.T]).T
      # ls = np.concatenate([ls]*3,-1).copy()
      # frame2 = cv2.addWeighted(frame2/255,1,ls,1,0,dtype=cv2.CV_32F)
      # frame= (frame2*255).clip(0,255).astype('uint8')

      self.update()
      return frame
