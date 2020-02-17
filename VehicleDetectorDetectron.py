import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2
import numpy as np

import pickle

BBOX_SCORE_THRESH = .5

class VehicleDetectorDetectron:
  wantsRGB = False
  def __init__(self):
    cfg = get_cfg()

    self.thangs = set()

    #cfg.MODEL.DEVICE = 'cpu'

    # See https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    self.predictor = DefaultPredictor(cfg)

  # Returns torch.tensor of bounding boxes for frame
  def getFeatures(self, frame):
    vehicleFeatures = self.predictor(frame)
    # See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    instances = vehicleFeatures['instances']
    keepScores = instances.scores > BBOX_SCORE_THRESH
    scores = instances.scores[keepScores]
    boxes = instances.pred_boxes[keepScores]
    classes = instances.pred_classes[keepScores]
    classesnpy = classes.cpu().numpy()
    masks = instances.pred_masks[keepScores]
    boundaries = self._getMaskBoundaries(masks)


    boxes = boxes.tensor.cpu().numpy()
    scores = [classesnpy[i] for i in range(len(boxes))]
    return boxes, scores, boundaries

  def _getMaskBoundaries(self, masks):
    boundaries = []
    for mask in masks:
      contour, hierarchy = cv2.findContours(np.uint8(mask.cpu().numpy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      boundaries.append(contour)
    return boundaries