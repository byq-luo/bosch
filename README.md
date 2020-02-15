# BUGS

### MED PRIO

* Video list bug

      1. Start app with TESTING=True
      2. Sort the video list by name or score column
      3. Open folder
      4. Notice that many video names are missing

### LOW PRIO

* Processing does not stop when app is closed

      1. Start app with TESTING=True
      2. Open folder
      3. Close app
      4. Notice that app is still processing videos


# TODO

### HIGHEST PRIO
* Get classifier working.

* Only return top 5 bounding boxes sorted by area?

* Sometimes a bounding box is given that covers the dash of the host vehicle. This box has width ~= width of screen. Do not return this bbox.

* Filter object detections if bounding boxes are too small?

* Given predictedLabels and groundTruth labels, how do we actually compute an accuracy?

### MED PRIO
* Batch process images instead of processing them one by one.

* Only return bounding boxes for vehicles.

* Labels are not parsed from disk properly or represented in memory properly. A label should have type similar to tuple(str, float)

* We load any and every video from disk. Instead, in TRAINING mode do not load vids that are missing ground truth labels

* Score in video list should be something like

      conf = DataPoint.aggregatePredConfidence
      score = conf is not None ? conf : '?'

* DataPoint.aggregatePredConfidence should not be assigned randomly. Should be assigned somewhere in Classifier.

* Decide if we should just blindly process videos or only process videos that have not predictedLabels in memory or on disk.

### LOWER PRIO

* Could learn a classifier to determine if car in bounding box is facing towards or away from us. Or if it is horizontal (like we are at an intersection). Or maybe there is some logical way to determine if the car is facing us or not.

* Increase performance of detection? Classifier runs the vehicle and lane detectors in an interleaved way. We should try running the vehicle detector first and the running the lane line detector. The lane line detector runs only on the cpu, so I wonder if we could get performance benefit by running it in parallel (in a separate process, might be a challenge)?
