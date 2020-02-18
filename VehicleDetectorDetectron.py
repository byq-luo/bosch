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

    boxes = instances.pred_boxes[keepScores]
    scores = instances.scores[keepScores]
    classes = instances.pred_classes[keepScores]
    masks = instances.pred_masks[keepScores]

    envelopes = self._getEnvelopes(masks)

    boxes = boxes.tensor.cpu().numpy()
    boxes = [boxes[i] for i in range(len(boxes))]
    classes = classes.cpu().numpy()
    classes = [classes[i] for i in range(len(classes))]
    return boxes, envelopes, scores, classes

  def _getEnvelopes(self, masks):
    envelopes = []
    for mask in masks:
      contour, hierarchy = cv2.findContours(np.uint8(mask.cpu().numpy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      envelopes.append(contour)
    return envelopes
