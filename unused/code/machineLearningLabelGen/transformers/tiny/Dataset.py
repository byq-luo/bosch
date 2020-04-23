import os, pickle, bisect, random
import numpy as np
import torch

# TODO will adding more points from the lane lines help?
# TODO can we use pixel values from the image somehow? what about using a 16*32 scaled version of the frames?
# TODO can we use more information from the ERFNET prob maps? using just those may be enough to predict all the labels

AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'BLANK']

PREDICT_EVERY_NTH_FRAME = 30
WINDOWWIDTH = 30*30
assert(WINDOWWIDTH % PREDICT_EVERY_NTH_FRAME == 0)

def clamp(l, h, x):
  if x < l: return l
  if x > h: return h
  return x

class Vehicle:
  def __init__(self, vehicle):
    objId, x1, y1, x2, y2 = vehicle
    self.objId = objId
    self.point = (x1,y1,x2,y2)
    self.maxAge = None
    self.ageAtFrame = None
    self.percentLifeComplete = None
    self.width = None
    self.height = None
    self.area = None
    self.inHostLane = None
    self.inLeftLane = None
    self.inRightLane = None
    self.avgSignedDistToLeftLane = None
    self.avgSignedDistToRightLane = None
    self.centroidProbLeft = None
    self.centroidProbRight = None
  def unpack(self):
    #assert(self.maxAge is not None)
    #assert(self.ageAtFrame is not None)
    #assert(self.percentLifeComplete is not None)
    #assert(self.width is not None)
    #assert(self.height is not None)
    #assert(self.area is not None)
    #assert(self.inLeftLane is not None)
    #assert(self.inHostLane is not None)
    #assert(self.inRightLane is not None)
    #assert(self.avgSignedDistToLeftLane is not None)
    #assert(self.avgSignedDistToRightLane is not None)
    #assert(self.centroidProbLeft is not None)
    #assert(self.centroidProbRight is not None)
    return (self.objId,
            self.maxAge,
            self.ageAtFrame,
            self.width,
            self.height,
            self.area,
            self.percentLifeComplete,
            self.inHostLane,
            *self.point,
            self.avgSignedDistToLeftLane,
            self.avgSignedDistToRightLane,
            self.centroidProbLeft,
            self.centroidProbRight,
            self.inLeftLane,
            self.inRightLane)

class Dataset():
  def load(self, path):
    with open(path, 'rb') as f:
      (self.data, self.offsets, self.memo, self.stats) = pickle.load(f)

  def save(self, path):
    with open(path, 'wb') as file:
      p = pickle.Pickler(file)
      p.fast = True
      p.dump((self.data, self.offsets, self.memo, self.stats))

  def getStats(self):
    return self.stats

  def __init__(self, featuresFilePaths=None, stats=None, loadPath=None):
    if loadPath is not None:
      self.load(loadPath)
      return
    self.data = []
    self.offsets = []
    self.memo = []
    self.stats = None
    #self.i2path = []

    for path in featuresFilePaths:
      data = processFeatures(path)
      if data is None:
        continue
      (data,counts,msg) = data
      self.data.extend(data)
      self.offsets.extend(counts)
      #if len(counts):
      #  self.i2path.append((counts[-1],path))
      print(msg)

    self.offsets = np.cumsum(self.offsets)
    print('\tNumSequences:',self.offsets[-1])

    # Calculate stats
    if stats is None:
      allxs,allys  = [],[]
      for ((xs,_),ys) in self.data:
        allxs.extend(xs)
        allys.extend(ys)
      bigboy = torch.cat(allxs)
      classCounts = dict()
      for label in range(len(AllPossibleLabels)):
        classCounts[label] = classCounts.get(label, 0) + allys.count(label)
      stats = (bigboy.mean(dim=0), bigboy.std(dim=0), classCounts)
      del bigboy, allxs
    self.stats = stats

    # Standardize data
    mean, std = self.stats[:2]
    for ((xs,_),_) in self.data:
      for x in xs:
        x -= mean
        x /= std

    # Memoize to avoid tons of bisect calls during training
    self.memo = {}
    for idx in range(self.offsets[-1]):
      index = bisect.bisect(self.offsets, idx)
      idxNew = idx - self.offsets[index-1] if index > 0 else idx
      self.memo[idx] = (idxNew,index)

  def __len__(self):
    return self.offsets[-1]

  def __getitem__(self, idx):
    # Where in self.data is the window we want?
    (idx, index) = self.memo[idx]

    ((xs,xlengths), ys) = self.data[index]

    # print(self.offsets)
    # print(index)
    # print(idx)
    # print(len(xs))
    # print(len(xlengths))
    # print(len(ys))

    xs = xs[idx:idx+WINDOWWIDTH]
    xlengths = xlengths[idx:idx+WINDOWWIDTH]
    ys = ys[idx//PREDICT_EVERY_NTH_FRAME:idx//PREDICT_EVERY_NTH_FRAME+WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME]

    #assert(len(xs) == WINDOWWIDTH)
    #assert(len(xlengths) == WINDOWWIDTH)
    #assert(len(ys) == WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)

    return ((xs,xlengths), ys)

def processFeatures(featuresPath):
  labelsPath = featuresPath.replace('../features/', '../labels/').replace('_m0.pkl', '_labels.txt')
  if not os.path.isfile(labelsPath):
    return None

  data = []
  counts = []
  msgs = 'Processing:' + featuresPath + ' '

  # Load features
  with open(featuresPath, 'rb') as file:
    features = list(pickle.load(file))

  # Get number of frames in video
  _, _, lines, _, vehicles, _ = features
  numFramesInVideo = len(lines)
  msgs += '\tNumFrames:' + str(numFramesInVideo) + ' '

  preprocessFeatures(features, numFramesInVideo)

  # Get labels, assumes labels file is sorted by time
  frameNum2LabelId = dict()
  with open(labelsPath) as file:
    lines = [line.rstrip('\n') for line in file.readlines()]
    prevLabel = None
    prevLabelTime = None
    prevLabelSkipped = False
    for line in lines:
      label, labelTime = line.split(',')
      label = label.split('=')[0]
      labelTime = float(labelTime)

      # Filter some bad labels
      if label in ['barrier','objTurnOff','end']:
        prevLabel = label
        continue
      if prevLabel and prevLabel == 'barrier' and label == 'rightTO':
        prevLabel = 'rightTO'
        continue
      if prevLabel and prevLabel == 'rightTO' and label == 'rightTO':
        prevLabel = 'rightTO'
        continue
      if prevLabel and prevLabel == 'objTurnOff' and label == 'evtEnd':
        prevLabel = None
        continue
      if prevLabel and prevLabel == 'end' and label == 'evtEnd':
        prevLabel = None
        continue
      if prevLabelSkipped and label == 'evtEnd':
        prevLabelSkipped = False
        continue

      if label not in AllPossibleLabels:
        print('.'+label+'.')
        exit(1)

      # What would the frame number be for this label if we accepted it?
      if prevLabelTime and (prevLabelTime%300) > (labelTime%300):
        frameNumber = ((labelTime-prevLabelTime)%300 + (prevLabelTime%300)) * 30
      else:
        frameNumber = (labelTime%300)*30

      if label in ['rightTO', 'cutin', 'cutout']:
        t=int(frameNumber)
        c = [len(vehicles[tp])
          for tp in range(    max(0,t-60),  min(len(vehicles)-1,t+61)    )]
        e = [int(x==0) for x in c]
        #if len(c) and sum(e)/len(e) > .9:
        #  print(label,end='')
        #  print(c)
        if sum(e) / len(e) > .8:
          print(label,e)
          prevLabel = None
          prevLabelSkipped = True
          continue

      prevLabel = label
      prevLabelTime = labelTime

      # Accept this label
      frameNumber = int(frameNumber / PREDICT_EVERY_NTH_FRAME + .5) * PREDICT_EVERY_NTH_FRAME
      while frameNumber in frameNum2LabelId:
        frameNumber += PREDICT_EVERY_NTH_FRAME
      if frameNumber > numFramesInVideo:
        break
      frameNum2LabelId[frameNumber] = AllPossibleLabels.index(label)
  msgs += '\tNumLabels:'+str(len(frameNum2LabelId)) + ' '

  # Convert label times to disjoint intervals
  intervals = []
  prvlow,prvhigh = None, None
  for frameNum in sorted(frameNum2LabelId):
    low = clamp(0,numFramesInVideo,frameNum - WINDOWWIDTH//3*2)
    high = clamp(0,numFramesInVideo - numFramesInVideo % PREDICT_EVERY_NTH_FRAME,frameNum + WINDOWWIDTH//3*2)
    if high - low < WINDOWWIDTH:
      continue
    if prvlow == None:
      prvlow,prvhigh = [low,high]
    elif low <= prvhigh:
      prvhigh = high
    else:
      intervals.append((prvlow,prvhigh))
      prvlow,prvhigh = low,high
  if prvlow is not None:
    intervals.append((prvlow,prvhigh))
  msgs += '\tNumIntervals:' + str(len(intervals)) + ' '

  # Create tensors for intervals
  for low,high in intervals:
    assert(low % PREDICT_EVERY_NTH_FRAME == 0)
    assert(high % PREDICT_EVERY_NTH_FRAME == 0)
    ((xs, xlengths), ys) = getDataForInterval(low,high,features,frameNum2LabelId)
    assert(len(xs) // PREDICT_EVERY_NTH_FRAME == len(ys))
    data.append(((xs, xlengths), ys))
    counts.append(len(xlengths) - WINDOWWIDTH + 1)
  return (data,counts,msgs)

def preprocessFeatures(features, numFramesInVideo):
  (_, _, lines, lanescores, vehicles, boxavglaneprobs) = features

  # Do a few things here;
  # Add a feature to describe the age in number frames of the box.
  # Filter boxes that do not exist in the video for long enough to matter.
  # Add a feature to describb how far and on which side of a lane the vehicle is.
  # This loop will also convert all the vehicles into Vehicle objects which
  # are infinitely easier to work with than raw tuples. This feature engineering
  # could have been done in Classifier.py but, well, we didn't do it and so we'll
  # do it here instead of recomputing features.
  objFirstFrame = dict()
  maxAge = dict()
  for i in range(numFramesInVideo):
    vehiclesInFramei = vehicles[i]
    if len(boxavglaneprobs[i]) != 2:
      boxavglaneprobs[i] = [[],[]]
    centroidProbsLeft = boxavglaneprobs[i][0]
    centroidProbsRight = boxavglaneprobs[i][1]
    for j,vehicleTuple in enumerate(vehiclesInFramei):
      v = Vehicle(vehicleTuple)
      v.centroidProbLeft = centroidProbsLeft[j]
      v.centroidProbRight = centroidProbsRight[j]

      # Add age at frame feature
      if v.objId not in objFirstFrame:
        objFirstFrame[v.objId] = i
      v.ageAtFrame = i - objFirstFrame[v.objId]
      maxAge[v.objId] = max(maxAge.get(v.objId, 0), v.ageAtFrame)

      vehiclesInFramei[j] = v

  # Add max age feature, filter the boxes, and lane info
  for i in range(numFramesInVideo):
    if len(lines[i]) != 2:
      lines[i] = [[],[]]
    leftLane, rightLane = lines[i]
    if len(leftLane) == 0: leftLane = [(0,0,0,0)]
    if len(rightLane) == 0: rightLane = [(0,0,0,0)]
    leftys = [y1 for (x1,y1,x2,y2) in leftLane]
    rightys = [y1 for (x1,y1,x2,y2) in rightLane]
    vehiclesInFramei = vehicles[i]
    newVehiclesInFramei = []
    for v in vehiclesInFramei:
      if maxAge[v.objId] >= 10:
        # vehiclesKept.add(v.objId)

        # Add age info
        v.maxAge = maxAge[v.objId]
        v.percentLifeComplete = v.ageAtFrame / v.maxAge

        # Add lane position info
        x1,y1,x2,y2 = v.point
        midX = (x1 + x2) / 2
        indxa = bisect.bisect(leftys, y2)
        indxb = bisect.bisect(rightys, y2)
        if indxa < len(leftys):
            v.inLeftLane = 1 if midX < (leftLane[indxa][0] + leftLane[indxa][2])/2 else 0
        else:
            v.inLeftLane = 0
        if indxb < len(rightys):
            v.inRightLane = 1 if midX > (rightLane[indxb][0] + rightLane[indxb][2])/2 else 0
        else:
            v.inRightLane = 0
        v.inHostLane = 1 if (not v.inLeftLane and not v.inRightLane) else 0
        v.avgSignedDistToLeftLane = (np.array([(x1+x2)/2 for (x1,y1,x2,y2) in leftLane]) - midX).mean()
        v.avgSignedDistToRightLane = (np.array([(x1+x2)/2 for (x1,y1,x2,y2) in rightLane]) - midX).mean()

        v.width = x2-x1
        v.height = y2-y1
        v.area = v.width * v.height

        v.objId %= 200

        newVehiclesInFramei.append(v)
    vehicles[i] = newVehiclesInFramei

  # Take every 8th point on the line
  for i in range(numFramesInVideo):
    if len(lines[i]) != 2:
      lines[i] = [[],[]]
    left,right = lines[i]
    left = left[::8][:20]
    right = right[::8][:20]
    if len(left) < 20:
      left += [(0,0,0,0)] * (20 - len(left))
    if len(right) < 20:
      right += [(0,0,0,0)] * (20 - len(right))
    left = [x1 for (x1,y1,x2,y2) in left]
    right = [x1 for (x1,y1,x2,y2) in right]
    lines[i] = (left,right)

def getDataForInterval(low, high, features, frameNum2LabelId):
  xs, ys = [], []

  # YOLOboxes, YOLOboxscores, lines, lanescores, vehicles, boxavglaneprobs
  (_, _, lines, lanescores, vehicles, _) = features

  # Only look at data from this interval of frames
  vehicles = vehicles[low:high]
  lanescores = lanescores[low:high]
  lines = lines[low:high]

  # for the jth frame in the interval
  for j, (vehicles_, laneScores_, lines_) in enumerate(zip(vehicles, lanescores, lines)):
    laneScores_ = [float(score.cpu().numpy()) for score in laneScores_]
    if len(laneScores_) != 4:
        laneScores_ = [0,0,0,0]
    leftLane = lines_[0]
    rightLane = lines_[1]

    isVideoBegin = low + j < 60

    features_xs = []
    # descending sort by percent life completed
    for vehicle in sorted(vehicles_, key=lambda v:-v.percentLifeComplete):
      tens = torch.tensor([[float(isVideoBegin), *vehicle.unpack(), *laneScores_, *leftLane, *rightLane]], dtype=torch.float)
      features_xs.append(tens)
    # there may be lane changes even if there are no boxes in the frame
    if len(features_xs) == 0:
      tens = torch.tensor([[float(isVideoBegin)] + [0.]*18 + [*laneScores_, *leftLane, *rightLane]], dtype=torch.float)  # make sure there is always an input tensor
      features_xs.append(tens)

    features_xs = torch.cat(features_xs)
    xs.append(features_xs)
    if (low+j) % PREDICT_EVERY_NTH_FRAME == 0:
      if low+j in frameNum2LabelId:
        labelId = frameNum2LabelId[low+j]
        ys.append(labelId)
      else:
        ys.append(AllPossibleLabels.index('BLANK'))

  xlengths = list(map(len, xs))
  return ((xs, xlengths), ys)

if __name__ == '__main__':
  print('Loading data.')

  # Get paths to precomputed features
  filepaths = []
  for (dirpath, dirnames, filenames) in os.walk('../features'):
    filepaths.extend(dirpath + '/' + f for f in filenames)
  random.shuffle(filepaths)

  trainSize = int(len(filepaths)*.85)
  trainFiles = filepaths[:trainSize]
  testFiles = filepaths[trainSize:]

  print('Num training videos', len(trainFiles))
  print('Num testing videos', len(testFiles))

  # Convert the data into tensors
  trainData = Dataset(trainFiles)
  testData = Dataset(testFiles, stats=trainData.getStats())

  print('Num training examples', len(trainData))
  print('Num testing examples', len(testData))

  print('Saving datasets')
  trainData.save('trainDataset.pkl')
  testData.save('testDataset.pkl')
  print('Saved. Exiting.')
