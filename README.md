# BUGS

### MED PRIO

* Possibly len(dp.boundingBoxes) != video.totalNumFrames. Look in Classifier.py where we check 'if not isFrameAvail:'.

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

* Given predictedLabels and groundTruth labels, how do we compute an accuracy?

* Clicking 'Process Videos' while videos are processing would probably cause problems. (at best should queue up new videos)

### MED PRIO
* Batch process images instead of processing them one by one? (try to increase GPU utilization)

* Only return bounding boxes for vehicles.

* Decide if we should just blindly process videos or only process videos that have not predictedLabels in memory or on disk.

### LOWER PRIO

* Could learn a classifier to determine if car in bounding box is facing towards or away from us. Or if it is horizontal (like we are at an intersection).

* Increase performance of detection? Classifier runs the vehicle and lane detectors in an interleaved way. We should try running the vehicle detector first and the running the lane line detector. The lane line detector runs only on the cpu, so I wonder if we could get performance benefit by running it in parallel (in a separate process, might be a challenge)?
