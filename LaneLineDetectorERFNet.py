import torch
import torch.backends.cudnn as cudnn
import cv2
from erfnet.models.erfnet import ERFNet
import torch.nn.functional as F
import numpy as np


SMOOTHING = .8


class LaneLineDetector:
  def __init__(self):
    self._initDL()
    self.smoothTensor = None
    self.smooth = np.array(
        [[-227.64484193, 360.,         330.10717963, 225.],
         [130.319165,    360.,         348.9750931,  225.],
         [583.2080832,   360.,         369.10749053, 225.],
         [1113.74467232, 359.99999999, 390.73517945, 225.]])

  def _initDL(self):
    num_class = 5
    self.model = ERFNet(num_class)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(self.device)

    state_dict = torch.load('erfnet/trained/ERFNet_trained.tar')['state_dict']

    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = k[7:]  # remove `module.`
      new_state_dict[name] = v
    # load params
    self.model.load_state_dict(new_state_dict)

    cudnn.benchmark = True
    cudnn.fastest = True
    self.model.eval()

  def _linesFromProb(self, prob):
    # need to threshold before hough
    prob = (255*(prob > .05)).astype('uint8')

    hough = cv2.HoughLinesP(prob,
                            rho=20,
                            theta=np.pi/180,
                            threshold=50,
                            minLineLength=150,
                            maxLineGap=30)

    if hough is None:
      return

    line = np.average(hough, axis=0)

    x1, y1, x2, y2 = line[0]

    # Get slope and y interception of line
    if x1 == x2:
      slope = float('inf')
      y_intercept = 0.0
    else:
      slope = (y2 - y1) / (x2 - x1)
      y_intercept = slope * (0-x1) + y1

    # Line start from button end with half of the height of frame
    y1 = prob.shape[0]
    y2 = int(50)
    # Get x1, x2 according to y1, y2, slope and intersection
    x1 = int((y1-y_intercept)/slope)
    x2 = int((y2-y_intercept)/slope)

    line = np.array([x1, y1, x2, y2])

    # Account for the crop and scale in _getProbs
    line = line * 720/976 + np.array([0, 207, 0, 207])

    return line.astype('int')

  def getLines(self, frame):
    ll, l, r, rr, lllrrr = self._getProbs(frame)

    # Visualize
    llc = cv2.resize(ll, (720, 153), interpolation=cv2.INTER_LINEAR)
    lc = cv2.resize(l, (720, 153), interpolation=cv2.INTER_LINEAR)
    rc = cv2.resize(r, (720, 153), interpolation=cv2.INTER_LINEAR)
    rrc = cv2.resize(rr, (720, 153), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype('float32')
    zz = np.zeros(frame.shape[:2])
    for i,m in enumerate([llc, lc, rc, rrc]):
      z = np.zeros(frame.shape[:2])
      z[207:207+153] = 255 * m
      if i == 0: frame += np.dstack([z,zz,zz])
      elif i == 1: frame += np.dstack([z,z,zz])
      elif i == 2: frame += np.dstack([zz,z,zz])
      elif i == 3: frame += np.dstack([zz,zz,z])
    frame = np.clip(frame, 0, 254.99).astype('uint8')
    # return frame, []

    ll = self._linesFromProb(ll)
    l = self._linesFromProb(l)
    r = self._linesFromProb(r)
    rr = self._linesFromProb(rr)
    ret = []
    for i, line in enumerate([ll, l, r, rr]):
      if line is None:
        continue
      self.smooth[i] = self.smooth[i] * SMOOTHING + (1-SMOOTHING) * line
      ret.append(self.smooth[i])
    if not len(ret):
      return frame, []
    return frame, np.vstack(ret).astype('int')

  # TODO This should work for arbitrary frame size
  def _getProbs(self, frame):
    frame_orig = frame

    # This model expects images of width 976
    scale = 976 / 720
    h, w, _ = frame.shape
    w = int(w * scale + .5)
    h = int(h * scale + .5)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    # Crop the sky and the dash
    frame = frame[281:281+208]

    with torch.no_grad():
      PERMUTATION = (2, 0, 1)  # rotate right
      INV_PERMUTATION = (1, 2, 0)  # rotate left

      # Put into format # CxHxW
      frame = np.transpose(frame, PERMUTATION)

      # Apply transform defined by model
      input_mean = np.array(self.model.input_mean)[:, np.newaxis, np.newaxis]
      input_std = np.array(self.model.input_std)[:, np.newaxis, np.newaxis]
      frame = (frame - input_mean) / input_std

      # Send to GPU
      frame = torch.from_numpy(frame.astype('float32')).to(self.device)
      # Add empty batch dimension
      frame = frame.unsqueeze(0)

      # Run model
      input_var = torch.autograd.Variable(frame)
      # TODO we can get scores for lanes here
      # output is BxCxHxW = batch x channel x height x width
      # lane_scores is BxO = batch x numLanes(==4)
      output, lane_scores = self.model(input_var)

      output = F.softmax(output, dim=1)

      # Remove empty batch dimension
      output = output.squeeze(0)

      output = 3*output * output * output

      #if self.smoothTens is None:
      #  self.smoothTens = output
      #self.smoothTens = self.smoothTens * SMOOTHING + (1-SMOOTHING) * output
      #output = self.smoothTens

      prob_map = output.cpu().numpy()

    # OUTPUT FORMAT
    # prob_map[0] is the output for the ith image in the batch
    # prob_map[0,0,:,:] is the probability map for all 4 lane lines
    # prob_map[0,i,:,:] is the probability map for the ith lane line (1<=i<=4)
    # prob_of_lane is a probability that the ith lane exists (1<=i<=4) (i.e. gives a score)

    lllrrr = 1-prob_map[0][:, :, np.newaxis]
    ll = prob_map[1][:, :, np.newaxis]
    l = prob_map[2][:, :, np.newaxis]
    r = prob_map[3][:, :, np.newaxis]
    rr = prob_map[4][:, :, np.newaxis]

    #h,w,_ = ll.shape
    #w = int(w * 720 / 976)
    #h = int(h * 720 / 976)
    #ll = cv2.resize(ll, (w, h), interpolation=cv2.INTER_LINEAR)
    #l = cv2.resize(l, (w, h), interpolation=cv2.INTER_LINEAR)
    #r = cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR)
    #rr = cv2.resize(rr, (w, h), interpolation=cv2.INTER_LINEAR)

    return ll, l, r, rr, lllrrr
