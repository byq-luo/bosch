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
    self.pbox = box
    # self.estBoxCol = sampleBoxColor(box, frame)
    # self.outline = seg # a more unique fingerprint than bboxes
    self.velocity = np.array([0, 0])  # in screenspace


class VehicleTracker:
  def __init__(self):
    self.objs = []
    self.pots = []
    self.next_id = 0

  def getObjs(self, frame, boxes, boxscores):
    boxes = np.array(boxes)
    boxesTensor = torch.from_numpy(boxes)
    pairedBoxes = set()
    pairedObjs = set()
    if len(boxes) > 0:
      if len(self.objs) > 0:
        objs = self.objs
        # objTensor = torch.from_numpy(np.array([o.box for o in objs]))
        # IOUs = torchvision.ops.box_iou(boxesTensor, objTensor)
        distances = np.zeros((len(boxes), len(objs)))
        for i in range(len(objs)):
          for j in range(len(boxes)):
            b1Centroid = (boxes[j][:2] + boxes[j][2:]) / 2
            b2Centroid = (objs[i].box[:2] + objs[i].box[2:]) / 2
            diff = b1Centroid - b2Centroid
            distances[i,j] = np.linalg.norm(diff)

        values = {}
        for i in range(len(objs)):
          row = distances[i]
          mx = max(row)
          for j in range(len(boxes)):
            if j in pairedBoxes:
              continue
            if mx-row[j] < 5 and mx < 10:
              values[j] = values.get(j, []) + [i]
        for j, _is in values.items():
          objs[j].freq = min(24, objs[j].freq + 3)
          for i in _is:
            # update the obj box
            diff = boxes[i] - objs[j].box
            objs[j].velocity = objs[j].velocity * .8 + .2 * (diff[:2] + diff[2:]) / 2
            objs[j].box = objs[j].box * .7 + .3 * boxes[i]
            pairedBoxes.add(i)
            pairedObjs.add(j)

      notPairedBoxes = [b for i, b in enumerate(boxes) if i not in pairedBoxes]
      if len(self.pots) > 0 and len(notPairedBoxes) > 0:
        pots = self.pots
        potsTensor = torch.from_numpy(np.array([p.box for p in pots]))
        IOUs = torchvision.ops.box_iou(boxesTensor, potsTensor)

        values = {}
        for i in range(len(boxes)):
          if i in pairedBoxes:
            continue
          for j in range(len(pots)):
            b1Centroid = (boxes[i][:2] + boxes[i][2:]) / 2
            b2Centroid = (pots[j].box[:2] + pots[j].box[2:]) / 2
            diff = b1Centroid - b2Centroid
            # dist = (np.linalg.norm(diff) /(1+IOUs[i][j])).numpy()
            dist = np.linalg.norm(diff)
            if dist < 10:
              values[j] = values.get(j, []) + [i]
        for j, _is in values.items():
          pots[j].freq = min(24, pots[j].freq + 3)
          for i in _is:
            if i not in pairedBoxes:
              # update the obj box
              diff = boxes[i] - pots[j].box
              pots[j].velocity = pots[j].velocity * .8 + .2 * (diff[:2] + diff[2:]) / 2
              pots[j].box = pots[j].box * .7 + .3 * boxes[i]
              pairedBoxes.add(i)
              pairedObjs.add(j)

    # Check for potential new objects
    for i in range(len(boxes)):
      if i not in pairedBoxes:
        self.pots.append(Vehicle(boxes[i]))

    for i in range(len(self.objs)):
      self.objs[i].box += .6666*np.hstack([self.objs[i].velocity]*2)
    for i in range(len(self.pots)):
      self.pots[i].box += .6666*np.hstack([self.pots[i].velocity]*2)

    # Look for potential objs to promote
    if len(self.pots) > 0:
      toPromote = []
      toRemove = []
      for o in self.pots:
        o.freq -= 1.5
        if o.freq <= 0:
          toRemove.append(o)
        elif o.freq >= 15:
          toPromote.append(o)
      for o in toRemove:
        self.pots.remove(o)
      for o in toPromote:
        o.id = self.next_id
        self.next_id += 1
        self.pots.remove(o)
        self.objs.append(o)

    # Look for objects to remove
    if len(self.objs) > 0:
      toRemove = []
      for o in self.objs:
        o.freq -= 1.5
        if o.freq <= 0:
          toRemove.append(o)
      for o in toRemove:
        self.objs.remove(o)

    # boxes = [o.box for o in self.objs if o.id is not None and o.freq > 17]
    # ids = [o.id for o in self.objs if o.id is not None and o.freq > 17]
    boxes = [o.box for o in self.objs]
    ids = [o.id for o in self.objs]
    return boxes, ids
