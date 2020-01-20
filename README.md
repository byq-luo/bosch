Ideas:

    Support batch processing of videos
    Show learning/training statistics for video file in GUI

Questions for next meeting:

    Q: How are video files accessed by computers that would run the app?
      Guess: videos are stored in a shared network drive and can be accessed as normal

    Q: Would it be possible to give the app a root folder and then select for processing all videos under that root?
      Guess: yes

    Q: Where will the machine learning predictions be stored?
      Guess: for a given video file, store results as text file in same directory as video file

    Q: Should we have some kind of database?
      I think we could get away with not having a database at all.
      We could put all data related to a certain video into a folder in the same directory as the video.
      This data might include (labels & times & confidences, bounding box locations & sizes, line lanes).
      However, putting this data into a db might make it easier to access, filter, and do other analysis.

    Q: Should we implement a system for finding/correcting incorrect predictions?
      Guess: maybe? maybe they have a system for this already

    Q: Would it be possible to upload the data (or a subset of the data) to the cloud, Google Cloud Platform for example?

    Q: Does bosch have compute servers we can use for learning, if we need to do learning?