# TODO

*  make feature extraction work for arbitrary img sizes. To do this you should only need to modify LaneLineDetectorERFNet, VehicleDetectorYolo, and LabelGenerator.

* We could run a higher accuracy vehicle detector (detectron/mask-rcnn) every 30 frames or so and then use faster trackers from OpenCV. This would probably work good. Fine-tuning detectron2 for vehicle detection would probably give a nice performance boost.

* Batch process images instead of processing them one by one? (try to increase GPU utilization)

* Run more Classifier processes at once to try and increase GPU utilization?

* We do not adjust the label time based on the lengths of the videos in a video sequence.