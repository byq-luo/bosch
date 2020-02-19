# This would work differently than the other versions

#import cv2
#
#class VehicleTracker:
#  def __init__(self):
#    # initialize OpenCV's special multi-object tracker
#    self.trackers = cv2.MultiTracker_create()
#		self.tracker = cv2.TrackerKCF_create()
#		self.trackers.add(tracker, frame, box)
#  
#  def getObjs(self, boxes):
#    # grab the updated bounding box coordinates (if any) for each
#    # object that is being tracked
#    (success, boxes) = trackers.update(frame)
#