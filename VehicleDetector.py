# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class VehicleDetector:
  def __init__(self):
    cfg = get_cfg()

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well

    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    ## Set score_threshold for builtin models
    #cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    #cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    #cfg.freeze()
    

    # If there is no GPU available (iMacs) we can do feature extraction on the GPU
    #cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
    self.predictor = DefaultPredictor(cfg)

  def getFeaturesForFrame(frame):
    # This code will draw things to the image. It will draw 'everything' detectron2
    # detects, not just vehicles.

    # from detectron2.utils.visualizer import Visualizer
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # for output format description

    # print(outputs['sem_seg'].to('cpu'))
    # x = outputs['instances'].to('cpu')
    # print(x)
    # print(x.pred_boxes)
    # print(x.scores)
    # print(x.pred_classes)
    # print(x.pred_masks)

    return self.predictor(frame)