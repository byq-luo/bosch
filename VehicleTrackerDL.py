from deepsort.deepsort import *

class VehicleTracker:
  def __init__(self):
    self.ds = deepsort_rbc()

  def getObjs(self, frame, boxes, boxes_scores):
    #Pass detections to the deepsort object and obtain the track information.
    tracks = self.ds.run_deep_sort(frame,boxes_scores,boxes)

    #Obtain info from the tracks.
    bboxes = []
    ids = []
    for track in tracks:
        bbox = track.to_tlbr() #Get the corrected/predicted bounding box
        id_num = str(track.track_id) #Get the ID for the particular track.
        features = track.features #Get the feature vector corresponding to the detection.
        bboxes.append(bbox)
        ids.append(id_num)
    return bboxes, ids
