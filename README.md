# BUGS

### MED PRIO

* Video list bug

      1. Start app with TESTING=True
      2. Sort the video list by name or score column
      3. Open folder
      4. Notice that many video names are missing

* Potential oom bug

### LOW PRIO

* Processing does not stop when app is closed

      1. Start app with TESTING=True
      2. Open folder
      3. Close app
      4. Notice that app is still processing videos


# TODO

### HIGHEST PRIO
* Clicking 'Process Videos' while videos are processing would probably cause problems. (at best should queue up new videos)

### MED PRIO
* Batch process images instead of processing them one by one? (try to increase GPU utilization)

* Decide if we should just blindly process videos or only process videos that have not predictedLabels in memory or on disk.
