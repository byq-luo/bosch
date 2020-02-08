import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

BBOX_SCORE_THRESH = .7

class VehicleDetectorDetectron:
  def __init__(self):
    cfg = get_cfg()

    # TODO which model do we want to use???
    # See https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    # ~0.25 real time
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # ~0.5 real time
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/rpn_R_50_C4_1x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/rpn_R_50_FPN_1x.yaml")

    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/rpn_R_50_C4_1x.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

    ## Set score_threshold for builtin models
    #cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    #cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    #cfg.freeze()

    # If there is no GPU available (iMacs) we can do feature extraction on the GPU
    # TODO
    #cfg.MODEL.DEVICE = 'cpu'
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
    self.predictor = DefaultPredictor(cfg)

  # Returns torch.tensor of bounding boxes for frame
  def getBoxes(self, frame):
    vehicleFeatures = self.predictor(frame)
    # See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    instances = vehicleFeatures['instances']
    boxes = instances.pred_boxes
    scores = instances.scores
    #classes = instances.pred_classes
    keepBoxes = boxes[scores > BBOX_SCORE_THRESH]

    # Should rename function if we decide to also return segmentations

    return keepBoxes
