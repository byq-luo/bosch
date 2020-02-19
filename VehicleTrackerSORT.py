from sort import *

class Vehicle:
  def __init__(self, box, _id):
    self.id = _id
    self.box = box

class VehicleTracker:
  def __init__(self):
    #create instance of SORT
    self.mot_tracker = Sort(max_age=60, min_hits=30) 

  def getVehicles(self, frame, boxes, boxscores):
    # update SORT
    inboxes = []
    for b,s in zip(boxes,boxscores):
      inboxes.append((*b,s))
    result = self.mot_tracker.update(np.array(inboxes))
    vehicles = []
    for x1,y1,x2,y2,i in result:
      vehicles.append(Vehicle((int(x1),int(y1),int(x2),int(y2)), int(i)))
    return vehicles
