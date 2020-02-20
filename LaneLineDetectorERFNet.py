import torch
import torch.backends.cudnn as cudnn
import cv2
from erfnet.models.erfnet import ERFNet
import torch.nn.functional as F
import numpy as np
import math as m


SMOOTHING = .9

# USE_HOUGH = False
# USE_POLY1 = False
# USE_POLY2 = False
# USE_SEGMENTS = True


def mix(a, b, m):
  return (a-b)*m + b


class LaneLineDetector:
  def __init__(self):
    self._initDL()
    self.smoothTensor = None
    self.smooth = np.array(
        [[-227.64484193, 360., 330.10717963, 225.],
         [130.319165,    360., 348.9750931,  225.],
         [583.2080832,   360., 369.10749053, 225.],
         [1113.74467232, 356., 390.73517945, 225.]])
    self.getInitial()
    self.velocity = np.array([[0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.]])
    self.lives = [0, 0, 0, 0]

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

  def _segmentsFromProb(self, i, prob):
    xs, ys = [], []
    height = prob.shape[0]
    prob = np.argmax(prob, axis=1)
    lowerbound = 20
    upperbound = 180
    if i == 0 or i == 3:
      upperbound = 150
    for x, y in enumerate(prob): # x = row, y = col
      if upperbound > x > lowerbound:
        if prob[x] > .1:
          xs.append(x)
          ys.append(y)
    return xs, ys

  def _eqnFromProb(self, i, prob, order=2):
    xs, ys = self._segmentsFromProb(i, prob)
    coeffs = np.polyfit(xs, ys, order)
    return coeffs, list(zip(xs, ys))

  def _lineFromProb(self, i, prob):
    coeffs, _ = self._eqnFromProb(i, prob, order=1)
    slope, y_intercept = coeffs # transpose space
    y_intercept = - y_intercept / slope
    slope = 1 / slope
    if not m.isfinite(slope):
      slope = float('inf')
      y_intercept = 0.0
    y1 = prob.shape[0]
    y2 = int(20)
    try:
      x1 = int((y1-y_intercept)/slope)
      x2 = int((y2-y_intercept)/slope)
    except:  # catch any inf/nan exceptions
      return None
    line = np.array([x1, y1, x2, y2])
    line = line * 720/976 + np.array([0, 207, 0, 207])
    return line

  def getLines(self, frame):
    ll, l, r, rr, scores = self._getProbs(frame)
    ll = self._lineFromProb(0,ll)
    l = self._lineFromProb(1,l)
    r = self._lineFromProb(2,r)
    rr = self._lineFromProb(3,rr)

    ret = []

    scores = [scores[1],scores[2]]
    for i, (line, score) in enumerate(zip([l, r], scores)):
      if score < .3:
        line = self.smooth[i]
        self.lives[i] -= 1
        if self.lives[i] <= 0:
          self.lives[i] = 0
          continue
      else:
        self.lives[i] = min(30, self.lives[i]+1)
      if self.lives[i] < 5:
        continue
      self.smooth[i] = mix(self.smooth[i], line, SMOOTHING)
      # diff = self.smooth[i] - line
      # vel = (diff[:2] + diff[2:]) / 2
      # self.smooth[i] += .5 * np.hstack([self.velocity[i]]*2)
      # self.velocity[i] = mix(self.velocity[i], vel, .6)
      ret.append((self.smooth[i].astype('int'), i))

    #lines = [l,r]
    #scores = [scores[1],scores[2]]
    #ids = [1,2]
    #for prob, score, i in zip(lines, scores, ids):
    #  if score < .3:
    #    self.lives[i] -= 1
    #    if self.lives[i] <= 0:
    #      self.lives[i] = 0
    #    continue
    #  else:
    #    self.lives[i] = min(30, self.lives[i]+1)
    #  if self.lives[i] < 5:
    #    continue
    #  points = self._segmentsFromProb(i, prob)
    #  if points is None:
    #    continue
    #  xs,ys = points
    #  points = []
    #  for (x1,y1,x2,y2) in zip(ys,xs,ys[1:],xs[1:]):
    #    x1 = int(x1*720/976)
    #    x2 = int(x2*720/976)
    #    y1 = int(y1*720/976+207)
    #    y2 = int(y2*720/976+207)
    #    self.ddd[(y1,i)] = mix(self.ddd.get((y1,i),0), x1, .98)
    #    self.ddd[(y2,i)] = mix(self.ddd.get((y2,i),0), x2, .98)
    #    x1 = self.ddd[(y1,i)]
    #    x2 = self.ddd[(y2,i)]
    #    points.append(tuple(map(int,(x1,y1,x2,y2))))
    #  ret.append((points, i))

    if not len(ret):
      return []

    return ret

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

      # if self.smoothTens is None:
      #  self.smoothTens = output
      #self.smoothTens = self.smoothTens * SMOOTHING + (1-SMOOTHING) * output
      #output = self.smoothTens

      prob_map = output.cpu().numpy()

    # OUTPUT FORMAT
    # prob_map[0] is the output for the ith image in the batch
    # prob_map[0,0,:,:] is the probability map for all 4 lane lines
    # prob_map[0,i,:,:] is the probability map for the ith lane line (1<=i<=4)
    # prob_of_lane is a probability that the ith lane exists (1<=i<=4) (i.e. gives a score)

    # lllrrr = 1-prob_map[0][:, :, np.newaxis]
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

    return ll, l, r, rr, lane_scores.cpu().numpy()[0]

  def getInitial(self):
    self.ddd = {(229, 1): 337.513942951734, (230, 1): 336.491712991789, (231, 1): 335.7361025770573, (232, 1): 333.0076909480966, (233, 1): 331.9793657422, (234, 1): 331.3566308682944, (235, 1): 329.68656087982515, (236, 1): 327.70340759502994, (237, 1): 327.2305739231362, (238, 1): 324.9084512056208, (239, 1): 323.0953465126762, (240, 1): 322.3112425065929, (241, 1): 319.16536028578815, (242, 1): 318.24380721463893, (243, 1): 317.91120211992785, (244, 1): 315.12668986968913, (245, 1): 312.80406095566616, (246, 1): 312.8263299068958, (247, 1): 309.04762729623457, (248, 1): 308.2549395342925, (249, 1): 308.3211710596783, (250, 1): 305.1393594744412, (251, 1): 303.3605485203334, (252, 1): 303.4781028699182, (253, 1): 299.3875938182747, (254, 1): 299.88133831541506, (255, 1): 295.9936795967692, (256, 1): 295.1437064421778, (257, 1): 295.4531664198842, (258, 1): 290.7994721681554, (259, 1): 289.6940026327563, (260, 1): 289.77534421307604, (261, 1): 285.9036256068208, (262, 1): 284.1843074563949, (263, 1): 284.50090349917286, (264, 1): 280.1262559359143, (265, 1): 278.4567048429918, (266, 1): 277.1785564581494, (267, 1): 274.2700039111821, (268, 1): 274.1798699956436, (269, 1): 270.52670190047866, (270, 1): 269.4077581960544, (271, 1): 268.77104982355445, (272, 1): 265.60722645509946, (273, 1): 264.3566978425667, (274, 1): 263.80424993345133, (275, 1): 259.7786935885541, (276, 1): 258.66898219798384, (277, 1): 257.8841916856643, (278, 1): 254.9948277381248, (279, 1): 253.7754742527143, (280, 1): 253.54695133476656, (281, 1): 249.73003389062487, (282, 1): 248.66078716953433, (283, 1): 245.46623974341998, (284, 1): 244.18723304635952, (285, 1): 243.63836353888902, (286, 1): 241.09130071850765, (287, 1): 240.09029176994392, (288, 1): 238.2289630576079, (289, 1): 235.1900301704227, (290, 1): 234.15574126790546, (291, 1): 232.94272158831626, (292, 1): 230.33272736947401, (293, 1): 229.13639767223262, (294, 1): 227.7158167827232, (295, 1): 225.4522257866413, (296, 1): 224.06453939300576, (297, 1): 222.71810889529058, (298, 1): 219.78112218093347, (299, 1): 218.3738340380806, (300, 1): 215.9943017105959, (301, 1): 214.9735231341368, (302, 1): 213.76158126808195, (303, 1): 211.27731995154312, (304, 1): 210.1813695556305, (305, 1): 208.16919748543984, (306, 1): 205.82136910171414, (307, 1): 204.69871424689566, (308, 1): 203.71497084619045, (309, 1): 200.82836000273878, (310, 1): 199.33159412195866, (311, 1): 198.05433483764853, (312, 1): 195.16695155217, (313, 1): 193.6360782548128, (314, 1): 191.84303082031724, (315, 1): 190.18654931968115, (316, 1): 188.39738415018687, (317, 1): 186.02111761505108, (318, 1): 184.95721088717625, (319, 1): 183.58101013673803, (320, 1): 181.11870029728638, (321, 1): 179.00496782801378, (322, 1): 178.19749744266926, (323, 1): 176.16102914839323, (324, 1): 174.83600886201538, (325, 1): 173.9996290064902, (326, 1): 171.38538712988716, (327, 1): 169.7091094740809, (328, 1): 167.47400202075457, (329, 1): 166.03186958674885, (330, 1): 164.4356642349681, (331, 1): 162.46922942409137, (332, 1): 160.9408868046798, (333, 1): 159.02400942310967, (334, 1): 156.56014562351587, (335, 1): 155.34692778141954, (336, 1): 154.17860092756757, (337, 1): 151.94379839978507, (338, 1): 149.24275785385151, (339, 1): 151.72845534887549, (229, 2): 376.9044021556762, (230, 2): 378.34558772187256, (231, 2): 379.2912710073877, (232, 2): 380.92910121602637, (233, 2): 381.2261443297949, (234, 2): 382.4496148311878, (235, 2): 386.1290512324452, (236, 2): 385.13410157679357, (237, 2): 388.2482266816313, (238, 2): 387.4857913776435, (239, 2): 387.74294003903924, (240, 2): 391.1001104116896, (241, 2): 390.7758111207027, (242, 2): 391.0127037320744, (243, 2): 394.0271758795284, (244, 2): 393.57886298877594, (245, 2): 393.9049913500054, (246, 2): 397.55395305985616, (247, 2): 397.1140755962936, (248, 2): 397.40488682848826, (249, 2): 400.45640456938685, (250, 2): 400.07152583362534, (251, 2): 400.42017082801755, (252, 2): 403.74436616516283, (253, 2): 403.4584375811318, (254, 2): 406.3602884261684, (255, 2): 405.171241367511, (256, 2): 406.3842017579052, (257, 2): 409.3700478985147, (258, 2): 408.40761414644896, (259, 2): 409.4624951824082, (260, 2): 412.6253640048607, (261, 2): 
411.55454238204857, (262, 2): 412.66175347320825, (263, 2): 415.3460831716497, (264, 2): 414.4910284931467, (265, 2): 415.47518835531537, (266, 2): 417.9795508229901, (267, 2): 417.8281744938261, (268, 2): 420.53175265486163, (269, 2): 420.33480309322323, (270, 2): 420.79409882950415, (271, 2): 423.1684893195921, (272, 2): 423.6467325708608, (273, 2): 424.08794606730595, (274, 2): 
426.86654351143414, (275, 2): 426.6787324796948, (276, 2): 427.5121049365036, (277, 2): 430.01742599851116, (278, 2): 430.18913188233086, (279, 2): 430.98302241757966, (280, 2): 433.35415357036254, (281, 2): 433.22170821212745, (282, 2): 435.9850416601694, (283, 2): 435.8393227875036, (284, 2): 436.5470447740299, (285, 2): 439.1785846967725, (286, 2): 439.0867343373766, (287, 2): 439.8594453847882, (288, 2): 442.447451939705, (289, 2): 442.53263001598697, (290, 2): 443.3087633530724, (291, 2): 445.79764461996535, (292, 2): 445.9010449863871, (293, 2): 446.84258995344186, (294, 2): 449.2058071401095, (295, 2): 449.1980562387578, (296, 2): 450.2628684798223, (297, 2): 452.5544802556117, (298, 2): 452.9197035136068, (299, 2): 455.14553384658274, (300, 2): 455.78016755456565, (301, 2): 456.4453270948732, (302, 2): 458.3669676424506, (303, 2): 459.20507035913175, (304, 2): 460.20699428639637, (305, 2): 462.5218290384527, (306, 2): 463.10075452773225, 
(307, 2): 463.63641646212244, (308, 2): 465.9934999582673, (309, 2): 466.51721610179936, (310, 2): 467.32572255959644, (311, 2): 469.62453362012144, (312, 2): 470.2279394819564, (313, 2): 472.53474000790266, (314, 2): 472.9394677871112, (315, 2): 473.9774535923666, (316, 2): 476.16489866587295, (317, 2): 476.74609711530763, (318, 2): 477.5377890375793, (319, 2): 479.27266219435694, (320, 2): 480.0340040785092, (321, 2): 480.87642894364484, (322, 2): 482.9867624601985, (323, 2): 483.7402172158461, (324, 2): 484.32735789076224, (325, 2): 485.9193075276006, (326, 2): 486.71694369292027, (327, 2): 488.5652864676946, (328, 2): 489.5598849379397, (329, 2): 490.4513197245651, (330, 2): 492.14928023913905, (331, 2): 492.63408740343783, (332, 2): 493.39007957715665, 
(333, 2): 495.31080946973026, (334, 2): 496.7659808191154, (335, 2): 497.53180618592603, (336, 2): 499.48015377456, (337, 2): 499.8749822596375, (338, 2): 500.4708310176361, (339, 2): 504.8571778416241}



# # Visualize
# llc = cv2.resize(ll, (720, 153), interpolation=cv2.INTER_LINEAR)
# lc = cv2.resize(l, (720, 153), interpolation=cv2.INTER_LINEAR)
# rc = cv2.resize(r, (720, 153), interpolation=cv2.INTER_LINEAR)
# rrc = cv2.resize(rr, (720, 153), interpolation=cv2.INTER_LINEAR)
# frame = frame.astype('float32')
# zz = np.zeros(frame.shape[:2])
# for i,m in enumerate([llc, lc, rc, rrc]):
#   z = np.zeros(frame.shape[:2])
#   z[207:207+153] = 255 * m
#   if i == 0: frame += np.dstack([z,zz,zz])
#   elif i == 1: frame += np.dstack([z,z,zz])
#   elif i == 2: frame += np.dstack([zz,z,zz])
#   elif i == 3: frame += np.dstack([zz,zz,z])
# frame = np.clip(frame, 0, 254.99).astype('uint8')
# # return frame, []
