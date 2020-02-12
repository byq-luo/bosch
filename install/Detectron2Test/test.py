# Some basic setup:
# Setup detectron2 logger
import detectron2
import numpy
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import _C
from Video import Video
import PIL.Image

cfg = get_cfg()
#cfg.MODEL.DEVICE = 'cpu'
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)



vid = Video("C:/Users/jgeng/PycharmProjects/bosch/video/Gen5_RU_2019-10-07_07-56-42-0001_m0.avi")

class runner:
    def __init__(self, video):
        self.vid = video
        self.run()

    def run(self):
        ret, frame = self.vid.get_frame()
        if frame is None:
            """video has played all the way through"""
            return
        if ret:
            image = PIL.Image.fromarray(frame)
            im = numpy.array(image)
            im = im[:,:,::-1].copy()
            #im = cv2.imread(image)
            outputs = predictor(im)

            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            src = v.get_image()[:, :, ::-1]
            cv2.imshow('', src)
            cv2.waitKey()
        #self.run()
        return


#im = cv2.imread("./input.jpg")
#im = cv2.imread("C:/Users/jgeng/PycharmProjects/bosch/video/Gen5_RU_2019-10-07_07-56-42-0001_m0.avi")


runner(vid)
'''
outputs = predictor(im)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

src = v.get_image()[:, :, ::-1]
cv2.imshow('',src)
cv2.waitKey()
'''