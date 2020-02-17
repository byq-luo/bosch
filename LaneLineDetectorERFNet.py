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
    self.smooth = None
    self.smoothL = None
    self.smoothR = None

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
  
  def _linesFromGhost(self, ghost):
    hough = cv2.HoughLinesP(ghost,
                            rho=20,
                            theta=np.pi/180,
                            threshold=50,
                            minLineLength=150,
                            maxLineGap=20)
    print(hough)
    print(type(hough))
    print(hough.shape)
    if hough is None:
      return np.zeros((1, 4), dtype=np.int)
    return np.average(hough)

  def getLines(self, frame):
    ll,l,r,rr = self._getGhosts(frame)

    llLines = self._linesFromGhost(ll)
    lLines = self._linesFromGhost(l)
    rLines = self._linesFromGhost(r)
    rrLines = self._linesFromGhost(rr)

    #INV_PERMUTATION = (1, 2, 0)  # rotate left
    ## VISUALIZATION
    #makeRGB = np.transpose(np.vstack([lineGhosts]*3), INV_PERMUTATION)
    #h,w,_ = makeRGB.shape
    #h = int(h * 720 / 976)
    #w = int(w * 720 / 976)
    #makeRGB = cv2.resize(makeRGB, (w, h), interpolation=cv2.INTER_LINEAR)
    #zeros = np.zeros(frame.shape, dtype='float32')
    #zeros[208:208+153,:] = makeRGB * 255
    #result = zeros + frame.astype('float32')
    #return np.minimum(np.ones_like(frame)*255, result).astype('int8'),[]

    # lineGhosts = (255 * (lineGhosts>.1)).astype('uint8')
    # lineGhostsRGB = np.transpose(np.vstack([lineGhosts]*3), INV_PERMUTATION).copy()
    # lineGhosts = np.transpose(lineGhosts, INV_PERMUTATION).copy()
    # lineGhosts = self._doPolygon(lineGhosts)
    # hough = cv2.HoughLinesP(lineGhosts,
    #                         rho=20,
    #                         theta=np.pi/180,
    #                         threshold=50,
    #                         minLineLength=150,
    #                         maxLineGap=20)
    # if hough is None:
    #   return lineGhostsRGB, np.zeros((2, 4),dtype=np.int)
    # return lineGhostsRGB, self._calculateLines(lineGhostsRGB, hough)
    # # return lineGhostsRGB, self._calculateLines(lineGhosts, hough)

  # TODO This should work for arbitrary frame size
  def _getGhosts(self, frame):
    scale = 976 / 720
    width = int(frame.shape[1] * scale + .5)
    height = int(frame.shape[0] * scale + .5)
    dsize = (width, height)
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_LINEAR)
    frame = frame[281:281+208, :976]
    frame_orig = frame
    with torch.no_grad():
      PERMUTATION = (2, 0, 1)  # rotate right
      INV_PERMUTATION = (1, 2, 0)  # rotate left

      frame = np.transpose(frame, PERMUTATION)
      input_mean = np.array(self.model.input_mean)[:, np.newaxis, np.newaxis]
      input_std = np.array(self.model.input_std)[:, np.newaxis, np.newaxis]
      frame = (frame - input_mean) / input_std
      frame = torch.from_numpy(frame.astype('float32')).to(self.device)
      frame = frame.unsqueeze(0)  # add empty batch dimension

      input_var = torch.autograd.Variable(frame)
      # TODO we can get scores for lanes here
      output, lane_scores = self.model(input_var)
      output = F.softmax(output, dim=1)

      if self.smooth is None:
        self.smooth = output
      self.smooth = self.smooth * SMOOTHING + (1-SMOOTHING) * (output*1.2)
      output = self.smooth

      prob_map = (1.-output).data.cpu().numpy()  # BxCxHxW = batch x channel x height x width
      # prob_of_lane = lane_scores.data.cpu().numpy() # BxO = batch x numLanes(==4)

      # OUTPUT FORMAT
      # prob_map[0] is the output for the ith image in the batch
      # prob_map[0,0,:,:] is the probability map for all 4 lane lines
      # prob_map[0,i,:,:] is the probability map for the ith lane line (1<=i<=4)
      #
      # prob_of_lane is a probability that the ith lane exists (1<=i<=4) (i.e. gives a score)

      return prob_map[:,1],prob_map[:,2],prob_map[:,3],prob_map[:,4],
