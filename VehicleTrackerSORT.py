from sort import *

class VehicleTracker:
  def __init__(self):
    #create instance of SORT
    self.mot_tracker = Sort() 

  def getObjs(self, frame, boxes, boxscores):
    # update SORT
    inboxes = []
    for b,s in zip(boxes,boxscores):
      inboxes.append((*b,s))
    result = self.mot_tracker.update(np.array(inboxes))
    ids = []
    newboxes = []
    for x1,y1,x2,y2,i in result:
      newboxes.append((x1,y1,x2,y2))
      ids.append(i)
    return newboxes,ids
