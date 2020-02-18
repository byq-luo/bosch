import torch
import torchvision
import numpy as np


# GOOD settings
# for detectron
# no box smoothing
# max freq of 20
# velocity smoother = .3

# Reducing velocity contribution to freq decay may help


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


# VELOCITY_SMOOTHING = .8
# BOX_SMOOTHING = .7

# VELOCITY_SMOOTHING = .3
# BOX_SMOOTHING = .5

VELOCITY_SMOOTHING = .3
BOX_SMOOTHING = .8


def mix(a, b, m):
  return (a-b)*m+b


class Vehicle:
  def __init__(self, box, seg=None, _id=None):
    self.id = _id
    self.freq = 3  # TODO give better name
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

  def getVehicles(self, frame, boxes, boxscores):
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
            b1Center = (boxes[i][:2] + boxes[i][2:]) * .5
            b2Center = (objs[j][:2] + objs[j][2:]) * .5
            diff = b1Center - b2Center
            dist = np.linalg.norm(diff) / IOUs[i, j]  # inf for non intersecting boxes
            if dist < 50:
              values.append((dist, i, j))
        values.sort()
        for dist, i, j in values:
          b2d[i] = min(b2d.get(i, 10000), dist)
          if i not in pairedBoxes and j not in pairedObjs:
            self.objs[j].freq = min(40, self.objs[j].freq + 20/(1+dist))
            # update the obj box
            diff = boxes[i] - self.objs[j].box
            self.objs[j].velocity = mix(self.objs[j].velocity, (diff[:2] + diff[2:]) / 2, VELOCITY_SMOOTHING)
            self.objs[j].box = mix(self.objs[j].box, boxes[i], BOX_SMOOTHING)
            pairedBoxes.add(i)
            pairedObjs.add(j)

      # Check for potential new objects
      for i in range(len(boxes)):
        if i not in pairedBoxes:
          # try to filter boxes that represent an existing obj but for some reason
          # did not get paired. i.e. try to reduce number of box creations
          if i not in b2d or 1000 >= b2d[i] >= 20:
            self.objs.append(Vehicle(boxes[i]))

      for i in range(len(self.objs)):
        if self.objs[i].id is not None:
          self.objs[i].box += .5*np.hstack([self.objs[i].velocity]*2)

    if len(self.objs) > 0:
      toRemove = []
      for o in self.objs:
        o.freq -= 1 + np.linalg.norm(o.velocity)  # fast things disappear
        if o.freq <= 0:
          toRemove.append(o)
        elif o.freq >= 18 and o.id is None:  # we are confident enough in this object so give it an id
          o.id = self.next_id
          self.next_id += 1
      for o in toRemove:
        self.objs.remove(o)

    return [o for o in self.objs if o.id is not None]
