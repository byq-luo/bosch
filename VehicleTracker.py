import torch
import torchvision
import numpy as np


# def sampleBoxColor(box, frame):
#   x1,y1,x2,y2 = box
#   w = int(x2-x1+.5)
#   h = int(y2-y1+.5)
#   imW = frame.shape[1]
#   imH = frame.shape[0]
#   N = 20
#   c = np.array([0.,0.,0.])
#   for i in range(N):
#     row = int(y1+.5)+random.randint(h//4,h//4+h//2-1)
#     col = int(x1+.5)+random.randint(w//4,w//4+w//2-1)
#     c += frame[min(imH-1,row), min(imW-1,col)] / 255 / 20
#   return c


class Vehicle:
  def __init__(self, box, seg=None, _id=None):
    self.id = _id
    self.freq = 3
    self.box = box
    # self.estBoxCol = sampleBoxColor(box, frame)
    # self.outline = seg
    self.velocity = np.array([0, 0])


class VehicleTracker:
  # A first approximation is to pair boxes that have maximum overlap
  # This may have issues when cars overlap in the image

  # I want to answer the question:
  #   for each box in frame i:
  #     which object (potential or current), from frame i+1, does it identify best with?

  def __init__(self):
    self.objs = []
    self.next_id = 0

  def getObjs(self, frame, boxes, boxscores):
    boxes = np.array(boxes)
    pairedBoxes = set()
    pairedObjs = set()
    b2d = {}
    if len(boxes) > 0:
      IOUs = []
      if len(self.objs) > 0:
        boxesTensor = torch.from_numpy(boxes)
        objs = np.array([o.box for o in self.objs])
        objTensor = torch.from_numpy(objs)
        IOUs = torchvision.ops.boxes.box_iou(boxesTensor, objTensor)

        values = []
        for i in range(len(boxes)):
          for j in range(len(self.objs)):
            b1Center = boxes[i][:2] * .5 + .5 * boxes[i][2:]
            b2Center = objs[j][:2] * .5 + .5 * objs[j][2:]
            diff = b1Center - b2Center
            dist = np.linalg.norm(diff) / IOUs[i,j] # inf for non intersecting boxes
            if dist < 50:
              values.append((dist,i,j))
        values.sort()
        for dist, i, j in values:
          b2d[i] = min(b2d.get(i,10000), dist)
          if i not in pairedBoxes and j not in pairedObjs:
            self.objs[j].freq = min(20, self.objs[j].freq + 20/(1+dist))
            # update the obj box
            diff = boxes[i] - self.objs[j].box
            # self.objs[j].velocity = self.objs[j].velocity * .8 + .2 * (diff[:2] + diff[2:]) / 2
            # self.objs[j].box = self.objs[j].box * .7 + .3 * boxes[i]
            self.objs[j].velocity = self.objs[j].velocity * .3 + .7 * (diff[:2] + diff[2:]) / 2
            self.objs[j].box = self.objs[j].box * .5 + .5 * boxes[i]
            self.objs[j].box = boxes[i]
            pairedBoxes.add(i)
            pairedObjs.add(j)

        ## IOUs = NxM, boxes = Nx4, objs = Mx4
        ## IOUs[i,j] = intersection area over union area of boxes[i] and objs[j]
        #IOUs = torchvision.ops.boxes.box_iou(boxesTensor, objTensor)
        #values = []
        #for i, r in enumerate(IOUs):
        #  for j, iou in enumerate(r):
        #    if iou < .3:
        #      continue
        #    values.append((iou, i, j))
        #values = sorted(values, reverse=True)
        #for iou, i, j in values:
        #  if i not in pairedBoxes and j not in pairedObjs:
        #    self.objs[j].freq = min(24, self.objs[j].freq + 5*iou)
        #    # update the obj box
        #    diff = boxes[i] - self.objs[j].box
        #    self.objs[j].velocity = self.objs[j].velocity * .8 + .2 * (diff[:2] + diff[2:]) / 2
        #    self.objs[j].box = self.objs[j].box * .4 + .6 * boxes[i]
        #    self.objs[j].box = boxes[i]
        #    self.objs[j].pbox = boxes[i]
        #    pairedBoxes.add(i)
        #    pairedObjs.add(j)

      # Check for potential new objects
      for i in range(len(boxes)):
        if i not in pairedBoxes:
          # try to filter boxes that are represent an existing obj but for some reason
          # did not get paired. i.e. try to reduce number of box creations
          if i not in b2d or 1000 >= b2d[i] >= 20:
            self.objs.append(Vehicle(boxes[i]))

      for i in range(len(self.objs)):
        if self.objs[i].id is not None:
          self.objs[i].box += .5*np.hstack([self.objs[i].velocity]*2)

    if len(self.objs) > 0:
      toRemove = []
      for o in self.objs:
        o.freq -= 1 + np.linalg.norm(o.velocity) # fast things disappear
        if o.freq <= 0:
          toRemove.append(o)
        elif o.freq >= 18 and o.id is None:  # we are confident enough in this object so give it an id
          o.id = self.next_id
          self.next_id += 1
      for o in toRemove:
        self.objs.remove(o)

    obs = [o.box for o in self.objs if o.id is not None]
    ids = [o.id for o in self.objs if o.id is not None]
    return obs, ids