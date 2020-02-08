# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3
from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from Video import Video

class VehicleDetectorYolo:
    def __init__(self):
        self.model_def="yolo/config/yolov3.cfg"
        self.weights_path="yolo/weights/yolov3.weights"
        self.conf_thres=0.8
        self.nms_thres=0.4
        self.batch_size=1
        self.n_cpu=0
        self.img_size=416

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)
        self.model.load_darknet_weights(self.weights_path)
        self.model.eval()  # Set in evaluation mode

    def getBoxes(self, frame):
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(frame)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        # Configure input
        img = img.unsqueeze_(0)
        input_imgs = Variable(img.to(self.device))

        # Get detections
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        if detections is not None:
          detections = rescale_boxes(detections[0], self.img_size, frame.shape[:2])
          #unique_labels = detections[:, -1].cpu().unique()
          #n_cls_preds = len(unique_labels)
          return detections[:,:4]
        return torch.tensor([[0,0,0,0]])