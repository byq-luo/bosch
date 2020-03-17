import torch
import torch.backends.cudnn as cudnn
import cv2
from erfnet.models.erfnet import ERFNet
import torch.nn.functional as F
import numpy as np
import math

def clamp(l,h,x):
  if x > h: return h
  if x < l: return l
  return x
def mix(a, b, m):
  return (a-b)*(1-m) + b

THRESHOLD = .2

TIMESMOOTH = .7
XSMOOTH = .7
XTRENDSMOOTH = .7
OUTPUTTIMESMOOTH = .8
OUTPUTXSMOOTH = .1

VOFF=18

class LaneLineDetector:
  def __init__(self):
    self._initDL()
    self.state = [None]*4
    self.output = [None]*4
    self.frameIndex = -1
    self.prevLaneLineComputation = [None]*6
    self.framesSinceDiscontinuous = 0

  def getLines(self, frame, vehicles):
    self.frameIndex += 1
    leftProbMap,rightProbMap,leftOutsideScore,leftScore,rightScore,rightOutsideScore = self.prevLaneLineComputation
    if self.frameIndex % 2 == 0:
      leftProbMap, rightProbMap, leftOutsideScore, leftScore, rightScore, rightOutsideScore = self._getProbs(frame)
      self.prevLaneLineComputation = leftProbMap, rightProbMap, leftOutsideScore, leftScore, rightScore, rightOutsideScore

    # How much probability does each bounding box cover?
    (vehicleLaneProbsL,vehicleLaneProbsR) = [],[]
    for vehicle in vehicles:
      x1, y1, x2, y2 = vehicle.box
      width = x2 - x1
      height = y2 - y1
      centroid = np.array([x1/2+x2/2, y1/2+y2/2])
      # Estimate probability in bounding box
      avgProbL, avgProbR = 0, 0
      for i in range(70):
        randPoint = np.random.randn(2) * np.array([width,height]) * .333 + centroid
        xp = int(clamp(0,975,976/720*randPoint[0]))
        yp = int(clamp(0,207,(randPoint[1]-207)*976/720))
        avgProbL += leftProbMap[yp,xp]
        avgProbR += rightProbMap[yp,xp]
      vehicleLaneProbsL.append(avgProbL / 70)
      vehicleLaneProbsR.append(avgProbR / 70)

    laneLines = []
    for probMap, score, _id in zip([leftProbMap,rightProbMap], [leftScore,rightScore], [1,2]):
      if score < .5:
        laneLines.append([])
        continue

      probMap = probMap[VOFF:-VOFF][::-1]
      maxs = np.max(probMap, axis=1)
      measurement = np.argmax(probMap, axis=1).astype('float32')
      if self.state[_id] is None:
        self.state[_id] = measurement.copy()

      # Count 'size' of discontinuities.
      count = 0
      px = measurement[0]
      for y,x in enumerate(measurement):
        if abs(x-px) <= 60 - y/8:
          px = x
        else:
          count += 1
      noise = count/len(measurement)

      # The measurement is too noisy.
      if noise > .1:
        newState = self.state[_id].copy()
        self.framesSinceDiscontinuous = 0
      else:
        # Try to remove some discontinuities
        newMeasurement = measurement
        for k in range(2):
          maxgradient = max(abs(np.gradient(newMeasurement)))
          newMeasurement = measurement.copy()
          grad = np.gradient(newMeasurement)
          # use previous state estimate
          for y,gx in enumerate(grad):
            if abs(gx) > 20:
              low = max(0,y-20)
              high = min(len(grad)-1,y+20)
              newMeasurement[low:high] = self.state[_id][low:high]
          px = newMeasurement[0]
          grad = np.gradient(newMeasurement)
          for y,x in enumerate(newMeasurement):
            if abs(x-px) >= 40 - y/8:
              px += grad[:y-1].mean()
              if math.isfinite(px):
                newMeasurement[y] = px
            else:
              px = newMeasurement[y]
        # If we made the curve more discontinuous then just use the original curve
        if max(abs(np.gradient(newMeasurement))) > maxgradient:
          newMeasurement = measurement

        # Check how different the measurement is from the previous state est
        diff = abs(self.state[_id] - newMeasurement).mean()
        if diff > 20: # Take the measurement to be the new state if the measurement is nearly perfect
          self.state[_id] = newMeasurement.copy()
          if self.framesSinceDiscontinuous < 7:
            self.output[_id] = newMeasurement.copy()
        newState = mix(self.state[_id], newMeasurement, 1-TIMESMOOTH)
        self.framesSinceDiscontinuous = min(self.framesSinceDiscontinuous+1,8)

      smoothx = newState[0]
      trendx = newState[1] - newState[0]

      # Output only smoothed versions of the internal state.
      smoothedState = newState.copy()
      for y,x in enumerate(newState):
        prevsmoothx = smoothx
        smoothx = mix(smoothx + trendx, x, 1-XSMOOTH)
        trendx = mix(trendx, smoothx-prevsmoothx, 1-XTRENDSMOOTH)
        smoothedState[y] = smoothx

      # Feedback loop
      # newState = mix(newState, smoothedState, .3)

      self.state[_id] = newState.copy()

      # Smooth our output in time
      if self.output[_id] is None:
        self.output[_id] = smoothedState
      self.output[_id] = mix(self.output[_id], smoothedState, 1-OUTPUTTIMESMOOTH)
      # Smooth the output a bit more
      smoothx = self.output[_id][0]
      for y,x in enumerate(self.output[_id]):
        prevsmoothx = smoothx
        smoothx = mix(smoothx + trendx, x, 1-OUTPUTXSMOOTH)
        trendx = mix(trendx, smoothx-prevsmoothx, 1-OUTPUTXSMOOTH)
        self.output[_id][y] = smoothx

      # Finally convert to the output format
      xs, ys = [], []
      for y,x in enumerate(self.output[_id]):
        try:
          xs.append(int(x*720/976))
          y = len(smoothedState) - y - 1
          ys.append(int((VOFF+y)*720/976+207))
        except:
          continue

      # Filter anything we are not confident about
      ox,oy=[],[]
      for y in range(len(xs)):
        if maxs[y] > THRESHOLD:
          ox.append(xs[y])
          oy.append(ys[y])
      xs,ys=ox,oy

      # If we filtered a lot then just ignore this lane
      if len(xs) < 85:
        xs,ys = [],[]

      laneLines.append(list(zip(xs, ys, xs[1:], ys[1:])))

    return laneLines,(leftOutsideScore,leftScore,rightScore,rightOutsideScore),(vehicleLaneProbsL,vehicleLaneProbsR)

  # TODO This should work for arbitrary frame size
  def _getProbs(self, frame):
    # This model expects images of width 976
    scale = 976 / 720
    h, w, _ = frame.shape
    w = int(w * scale + .5)
    h = int(h * scale + .5)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    # Crop the sky and the dash
    frame = frame[281:281+208]
    with torch.no_grad():
      PERMUTATION = (2, 0, 1) # Put channels first
      INV_PERMUTATION = (1, 2, 0) # Put channels last
      frame = np.transpose(frame, PERMUTATION)
      # Model expects a certain mean and std for inputs
      frame = (frame - self.input_mean) / self.input_std
      frame = torch.from_numpy(frame.astype('float32')).to(self.device)
      # Add empty batch dimension
      frame = frame.unsqueeze(0)
      input_variable = torch.autograd.Variable(frame)
      # output is BxCxHxW = batch x channel x height x width
      # lane_scores is BxO = batch x numLanes(==4)
      output, lane_scores = self.model(input_variable)
      output = F.softmax(output, dim=1)
      # Remove empty batch dimension
      output = output[0]
      # output[output<THRESHOLD]*=0.
      lane_scores = lane_scores[0]
      # output[0] is the probability map for all 4 lane lines
      # output[i] is the probability map for the ith lane line (1<=i<=4)
      # lane_scores[i] is a probability that the ith lane exists (1<=i<=4)

      ll = output[1]
      leftOutsideScore = lane_scores[0]
      rr = output[4]
      rightOutsideScore = lane_scores[3]

      l = output[2]
      leftScore = lane_scores[1]
      r = output[3]
      rightScore = lane_scores[2]
    return l.cpu().numpy(), r.cpu().numpy(), leftOutsideScore,leftScore,rightScore,rightOutsideScore

  def _initDL(self):
    num_class = 5
    self.model = ERFNet(num_class)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(self.device)
    state_dict = torch.load('erfnet/trained/ERFNet_trained.tar', map_location=self.device)['state_dict']
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = k[7:]  # remove `module.`
      new_state_dict[name] = v
    self.model.load_state_dict(new_state_dict)
    cudnn.benchmark = True
    cudnn.fastest = True
    self.model.eval()
    self.input_mean = np.array(self.model.input_mean)[:, np.newaxis, np.newaxis]
    self.input_std = np.array(self.model.input_std)[:, np.newaxis, np.newaxis]