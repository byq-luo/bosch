from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint, QTimer
from PyQt5.QtGui import QPainter, QImage

from Video import Video
from VideoOverlay import VideoOverlay
import time

#imports used only for vehicles and lane lines
'''
import numpy
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import PIL.Image
import lane
import cv2
'''

class VideoWidget(QWidget):
  def __init__(self, centralWidget):
    super().__init__()
    self.initUI()

    # only need these fo vehicles and lane lines
    '''
    self.cfg = get_cfg()
    #self.cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.predictor = DefaultPredictor(self.cfg)
    '''

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
    format(seconds, '.2f')
    timeString = str(minutes) + ":" + str(seconds)
    self.currentTimeLabel.setText(timeString)

  def updateSliderValue(self):
    currentPercent = int(100 * self.video.getFrameNumber() / self.video.getTotalNumFrames())
    self.slider.setValue(currentPercent)

  '''
  converts numpy array image to PyQT QImage
  source: https://emresahin.net/converting-numpy-image-to-qimage-14359-22392/
  '''
  '''
  def get_qimage(self, image: numpy.ndarray):
    assert (numpy.max(image) <= 255)
    image8 = image.astype(numpy.uint8, order='C', casting='unsafe')
    height, width, colors = image8.shape
    bytesPerLine = 3 * width

    image = QImage(image8.data, width, height, bytesPerLine,
                       QImage.Format_RGB888)

    image = image.rgbSwapped()
    return image
  '''

  def _drawImage(self, frame, qp):
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
    '''
    # uncomment this code to draw vehicles and lane lines

    size = (int(self.video.vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    kernel = 5
    canny = lane.do_canny(frame, kernel)
    polygon = lane.do_polygon(canny, size[0], size[1])

    hough = cv2.HoughLinesP(polygon, 1, numpy.pi / 180, 50, minLineLength=70, maxLineGap=10)
    # for line in hough:
    # x1, y1, x2, y2 = line[0]
    # cv2.line(frames, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("polygon", frames)

    lines = lane.calculate_lines(frame, hough)
    lines_visualize = lane.visualize_lines(frame, lines)
    output = cv2.addWeighted(frame, 0.6, lines_visualize, 1, 0.1)

    image = PIL.Image.fromarray(output)
    im = numpy.array(image)
    im = im[:, :, ::-1].copy()
    outputs = self.predictor(im)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    image = v.get_image()[:, :, ::-1]
    image = self.get_qimage(image)
    '''
    # end of code that draws vehicles and lane lines

    image = image.scaled(scaledWidth, scaledHeight)
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

