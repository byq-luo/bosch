import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F

class LaneLineDetector:
    def __init__(self):
        num_class = 5
        self.model = models.ERFNet(num_class)
        checkpoint = torch.load('erfnet/pretrained/ERFNet_pretrained.tar')
        torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
        cudnn.benchmark = True
        cudnn.fastest = True
        model.eval()

    def getLines(self, frame):
        frame = cv2.resize(frame, None, fx=976 * 1.0 / 1640, fy=208 * 1.0 / 350, interpolation=cv2.INTER_LINEAR)

        input_mean = self.model.input_mean
        input_std = self.model.input_std
        frame = (frame - input_mean) / input_std

        input_var = torch.autograd.Variable(frame, volatile=True)
        # compute output
        output, output_exist = self.model(input_var)
        # measure accuracy and record loss
        output = F.softmax(output, dim=1)
        pred = output.data.cpu().numpy() # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy() # BxO
