USE_PRECOMPUTED_FEATURES = False

MAKE_PRECOMPUTED_FEATURES = True

# Decoding the video takes a lot of time. So do not decode unless it is necessary.
SHOULD_LOAD_VID_FROM_DISK = not USE_PRECOMPUTED_FEATURES

assert(not (USE_PRECOMPUTED_FEATURES and MAKE_PRECOMPUTED_FEATURES))

# Do not use Classifier.py, just run the feature extraction in videowidget.py.
IMMEDIATE_MODE = False