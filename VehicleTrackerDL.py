from deepsort.deepsort import *
import cv2

class Vehicle:
  def __init__(self, box, _id):
    self.id = _id
    self.box = box
  def __repr__(self):
    return str(self.id) + '  ' + str(self.box)

class VehicleTracker:
  def __init__(self):
    self.ds = deepsort_rbc()

  def getVehicles(self, frame, boxes, boxes_scores):
    # VISUALIZE FINGERPRINTS!
    framec = frame / 255
    for b in boxes:
      x1,y1,x2,y2= b
      bb = (min(x1,x2),min(y2,y1),abs(x1-x2),abs(y2-y1))
      feat, _ = self.ds.extract_features_only(frame, bb)
      feat = feat.reshape((32,32))
      feat=cv2.resize(feat, (int(x2-x1),int(y2-y1)),interpolation=cv2.INTER_NEAREST)
      # feat = (feat - feat.min()) / (feat.max()-feat.min())
      feat = abs(feat)*5*2
      feat = np.pad(feat, ((int(y1),frame.shape[0]-int(y2)),(int(x1),frame.shape[1]-int(x2))),'constant')
      feat = np.array([feat.T]).T
      feat = np.concatenate([feat]*3,axis=-1)
      framec = cv2.addWeighted(framec, 1, feat, 1, 0,dtype=cv2.CV_32F)
    return (framec*255).clip(0,255).astype('uint8').copy()

    #Pass detections to the deepsort object and obtain the track information.
    tracks = self.ds.run_deep_sort(frame,boxes_scores,boxes)
    #Obtain info from the tracks.
    vehicles =[]
    for track in tracks:
        bbox = track.to_tlbr() #Get the corrected/predicted bounding box
        id_num = str(track.track_id) #Get the ID for the particular track.
        features = track.features #Get the feature vector corresponding to the detection.
        x1,y1,x2,y2 = bbox
        vehicles.append(Vehicle((int(x1),int(y1),int(x2),int(y2)), int(id_num)))
    return vehicles
