# yolo code from https://github.com/eriklindernoren/PyTorch-YOLOv3

import pickle
import numpy as np

class VehicleDetector:
    wantsRGB = True
    def __init__(self):
        self.frameNumber = 0
    
    # Sometimes YOLO detects the host vehicle's dash. This function removes that detected bounding box.
    def removeBadBox(self, boxes, screenWidth):
        if len(boxes) == 0:
            return boxes
        widths = abs((boxes[:,0] - boxes[:,2])) # TODO this is either always positive or always negative
        return boxes[widths / screenWidth < .8]
    
    #def keepTopTenAreaBoxes(self, boxes):
    #    areas = abs((boxes[:,0] - boxes[:,2]))*abs((boxes[:,1] - boxes[:,3]))
    #    return boxes[areas.argsort()[-3:][::-1]]

    def getFeatures(self, frame):
        boxes = [b for b in self.bboxes[self.frameNumber] if (b != [0,0,0,0]).all()]
        if type(boxes) == list: # TODO remove this crap
            boxes = np.array(boxes)
        # segmentations = self.seg[self.frameNumber]
        segmentations = []
        self.frameNumber = (self.frameNumber + 1) % len(self.bboxes)
        #return self.removeBadBox(bboxes, frame.shape[1]), segmentations
        # TODO !!!! give actual frame width
        boxes = self.removeBadBox(boxes, 700)
        #boxes = self.keepTopTenAreaBoxes(boxes)
        #boxes = self.keepVehiclesFacingUs(boxes)
        return boxes, segmentations

    def loadFeaturesFromDisk(self, featuresPath):
        assert(self.frameNumber == 0)
        with open(featuresPath, 'rb') as file:
            # self.bboxes, self.seg, _ = pickle.load(file)
            self.bboxes = pickle.load(file)[0]
            if type(self.bboxes[0]) == list: # TODO this can be removed once all the pkls contain the same obj types
                self.bboxes = self.bboxes[0]