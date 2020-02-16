# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3
from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2
import numpy as np

from Video import Video

MODEL_CFG_PATH = 'yolo/cfg/yolov3-spp.cfg'
MODEL_WEIGHTS_PATH = 'yolo/weights/ultralytics68.pt'


def eq(x, y, eps=.00001):
    return abs(x-y) < eps


class VehicleDetectorYolo:
    wantsRGB = True

    def __init__(self):
        self.model_def = MODEL_CFG_PATH
        self.weights = MODEL_WEIGHTS_PATH
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.batch_size = 1
        self.img_size = 608
        self.half = False

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)

        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        self.model.to(self.device).eval()

        # Half precision
        # half precision only supported on CUDA
        self.half = self.half and self.device.type != 'cpu'
        if self.half:
            self.model.half()

    def getFeatures(self, frame):
        # Padded resize
        img = letterbox(frame, new_shape=self.img_size)[0]
        img = img.transpose(2, 0, 1)  # reshape to 3x608x608
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Extract image as PyTorch tensor
        img = torch.from_numpy(img).to(self.device)

        # add empty dim to form batch of size 1?
        img = img.unsqueeze_(0)

        with torch.no_grad():
            detections = self.model(img)[0]
            if self.half:
                detections = detections.float()
            detections = non_max_suppression(detections, self.conf_thres, self.iou_thres)

        segmentations = []  # YOLO does not give segmentations
        if detections is not None and detections[0] is not None:
            detections = detections[0]
            # Rescale boxes from img_size to im0 size
            detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], frame.shape[:2]).round()

            # we only care about labels 2, 3, 5, 7. See https://github.com/ultralytics/yolov3/blob/master/data/coco.names
            #unique_labels = detections[:, -1].cpu().unique()
            #n_cls_preds = len(unique_labels)

            keepBoxes = []
            for x1, y1, x2, y2, score, clazz in detections.cpu().numpy():
                if eq(clazz, 2) or eq(clazz, 3) or eq(clazz, 5) or eq(clazz, 7):
                    keepBoxes.append((x1, y1, x2, y2))
            return np.array(keepBoxes), segmentations
        return np.array([]), segmentations
