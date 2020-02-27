import torch
import torch.backends.cudnn as cudnn
import cv2
from erfnet.models.erfnet import ERFNet
import torch.nn.functional as F
import numpy as np
import math as m


SMOOTHING = .93
THRESHOLD = .05


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

    # TODO this assumes that there is a cuda device avail
    # torch.load has a parameter that can fix this for us
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

  def _segmentsFromProb(self, prob):
    rs, cs = [], []
    indices = np.argmax(prob, axis=1)
    lowerbound = 23
    upperbound = 180
    for r, c in enumerate(indices):
      if upperbound > r > lowerbound:# and prob[r][c] > THRESHOLD:
        rs.append(r)
        cs.append(c)
    if len(rs) == 0:
      return None
    # coeffs = np.polyfit(rs, cs, deg=4)
    # cs = []
    # for r in rs:
    #   c = r**4*coeffs[0] + r**3*coeffs[1] + r**2*coeffs[2] + r*coeffs[3] + coeffs[4]
    #   cs.append(c)
    return cs, rs

  def getLines(self, frame):
    l, r, scores = self._getProbs(frame)
    ret = []

    lines = [l, r]
    scores = [scores[1], scores[2]]
    ids = [1, 2]
    for prob, score, i in zip(lines, scores, ids):
      if score < .3:
        self.lives[i] -= 1
        if self.lives[i] <= 0:
          self.lives[i] = 0
        continue
      else:
        self.lives[i] = min(30, self.lives[i]+1)
      if self.lives[i] < 5:
        continue
      points = self._segmentsFromProb(prob)
      if points is None:
        continue
      xs, ys = points
      points = []
      for (x1, y1, x2, y2) in zip(xs, ys, xs[1:], ys[1:]):
        x1 = int(x1*720/976)
        x2 = int(x2*720/976)
        y1 = int(y1*720/976+207)
        y2 = int(y2*720/976+207)
        self.ddd[(y1, i)] = mix(self.ddd.get((y1, i), 0), x1, SMOOTHING)
        self.ddd[(y2, i)] = mix(self.ddd.get((y2, i), 0), x2, SMOOTHING)
        x1 = self.ddd[(y1, i)]
        x2 = self.ddd[(y2, i)]
        points.append(tuple(map(int, (x1, y1, x2, y2))))
      ret.append((points, i))

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
      # output: torch.Tensor = output

      # Remove empty batch dimension
      output = output.squeeze(0)

      output = F.softmax(output, dim=0)

      # throw out ll and rr lanes
      output = output[2:4]

      l,r = output.cpu().numpy()
      lane_scores = lane_scores.cpu().numpy()[0]

    # OUTPUT FORMAT
    # prob_map[0] is the output for the ith image in the batch
    # prob_map[0,0,:,:] is the probability map for all 4 lane lines
    # prob_map[0,i,:,:] is the probability map for the ith lane line (1<=i<=4)
    # prob_of_lane is a probability that the ith lane exists (1<=i<=4) (i.e. gives a score)

    return l, r, lane_scores

  def getInitial(self):
    self.ddd = {(229, 1): 337.6286111937748, (230, 1): 336.59481701285296, (231, 1): 335.8080498335211, (232, 1): 333.17666983263524, (233, 1): 331.9376409057544, (234, 1): 331.28922824406783, (235, 1): 329.66188017611137, (236, 1): 327.65029960476085, (237, 1): 327.0816487892305, (238, 1): 324.7400962619016, (239, 1): 322.93469582990656, (240, 1): 322.33573833136234, (241, 1): 319.09005786022243, (242, 1): 318.2542040204773, (243, 1): 317.94666134211553, (244, 1): 315.0497885889748, (245, 1): 312.713854264906, (246, 1): 312.6433812581224, (247, 1): 308.80252819401204, (248, 1): 308.0183876434852, (249, 1): 307.9720068611327, (250, 1): 304.89803852465894, (251, 1): 303.0458802567962, (252, 1): 303.0133480650991, (253, 1): 299.23896769343156, (254, 1): 299.70301949249637, (255, 1): 295.62264351155204, (256, 1): 294.7376226485412, (257, 1): 294.8179812497026, (258, 1): 290.46165556899604, (259, 1): 289.31060321285264, (260, 1): 289.21530879100175, (261, 1): 285.6113310760113, (262, 1): 283.82111873217383, (263, 1): 283.86973615771734, (264, 1): 279.84313483733325, (265, 1): 278.1047416408923, (266, 1): 276.7761561700569, (267, 1): 273.9928036177062, (268, 1): 273.6487813672352, (269, 1): 270.1363834842951, (270, 1): 269.17746546487496, (271, 1): 268.37558423942585, (272, 1): 265.3439214355308, (273, 1): 264.1242942900284, (274, 1): 263.282779767393, (275, 1): 259.6873836217334, (276, 1): 258.44944886286777, (277, 1): 257.58071441445054, (278, 1): 254.74749314284412, (279, 1): 253.60155941621008, (280, 1): 253.17215699969506, (281, 1): 249.47177540837595, (282, 1): 248.37973970155502, (283, 1): 245.27956531582785, (284, 1): 243.98905300363757, (285, 1): 243.39836173350952, (286, 1): 240.88916176943525, (287, 1): 239.92934295687985, (288, 1): 237.9149710652031, (289, 1): 235.03318855680493, (290, 1): 233.9562518886717, (291, 1): 232.64757082309438, (292, 1): 230.22299713378365, (293, 1): 228.97733809693207, (294, 1): 227.56819982052343, (295, 1): 225.38865035261628, (296, 1): 223.94374423640585, (297, 1): 222.5723849959549, (298, 1): 219.73110194348754, (299, 1): 218.2386666494601, (300,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    1): 215.87060173426948, (301, 1): 214.80773247563386, (302, 1): 213.61441581885512, (303, 1): 211.08281887803773, (304, 1): 209.9829109922889, (305, 1): 208.05633937501068, (306, 1): 205.69052763175804, (307, 1): 204.5215813384455, (308, 1): 203.31138504001476, (309, 1): 200.65656857141366, (310, 1): 199.18055083466433, (311, 1): 197.8098109087612, (312, 1): 195.09158072580243, (313, 1): 193.43711692630558, (314, 1): 191.63060177747525, (315, 1): 190.0707483622218, (316, 1): 188.17833294504646, (317, 1): 185.81602530077706, (318, 1): 184.74948236848314, (319, 1): 183.2554156432891, (320, 1): 180.95885911344578, (321, 1): 178.92290048864618, (322, 1): 178.0457884006547, (323, 1): 176.00292794435668, (324, 1): 174.62327190343066, (325, 1): 173.70586760787336, (326, 1): 171.23653167669136, (327, 1): 169.30673048629922, (328, 1): 167.20509772310749, (329, 1): 165.9096646422122, (330, 1): 164.09443629878257, (331, 1): 162.24135574949983, (332, 1): 160.6499936027056, (333, 1): 158.69106720171362, (334, 1): 156.33601429384527, (335, 1): 155.15523638737233, (336, 1): 153.73157127164245, (337, 1): 151.65301552734408, (338, 1): 149.04676464279083, (339, 1): 151.92146665556123, (229, 2): 378.7824350704452, (230, 2): 378.31898161740173, (231, 2): 379.1799605750242, (232, 2): 380.71016928465485, (233, 2): 380.82352073207505, (234, 2): 382.0974395916424, (235, 2): 385.6332180772579, (236, 2): 384.8926179711728, (237, 2): 387.6370254172557, (238, 2): 387.13511281949565, (239, 2): 387.444097521273, (240, 2): 390.6877073597684, (241, 2): 390.437094174019, (242, 2): 390.6012891325465, (243, 2): 393.3973697009334, (244, 2): 393.232028109547, (245, 2): 393.61283680744447, (246, 2): 396.96848349602675, (247, 2): 396.7893125661146, (248, 2): 397.0096494159833, (249, 2): 399.8627247603104, (250, 2): 399.7450113049305, (251, 2): 400.0667994433941, (252, 2): 403.21615568942985, (253, 2): 403.0654142517175, (254, 2): 405.7172861068198,
                (255, 2): 404.8076106555048, (256, 2): 406.0293532124265, (257, 2): 408.81203277755134, (258, 2): 408.05373297683127, (259, 2): 409.06964514456297, (260, 2): 411.9617390145371, (261, 2): 411.20672261493064, (262, 2): 412.2358897860989, (263, 2): 414.7001851104991, (264, 2): 414.0993599769197, (265, 2): 415.0828676866149, (266, 2): 417.42990202889615, (267, 2): 417.4916436041819, (268, 2): 419.8619489372302, (269, 2): 419.93669878870156, (270, 2): 420.4561658607602, (271, 2): 422.5076439325268, (272, 2): 423.3027261194376, (273, 2): 423.76213253224216, (274, 2): 426.01535333984793, (275, 2): 426.2535828579153, (276, 2): 427.08008562204435, (277, 2): 429.3455663939063, (278, 2): 429.70256532714666, (279, 2): 430.44672535974746, (280, 2): 432.4152045868102, (281, 2): 432.73648789394, (282, 2): 435.18165750709466, (283, 2): 435.3383382494676, (284, 2): 436.07524231453505, (285, 2): 438.267791786281, (286, 2): 438.59595451238476, (287, 2): 439.4005273853498, (288, 2): 441.64345423644136, (289, 2): 442.0602391621859, (290, 2): 442.8683754388892, (291, 2): 445.023121158447, (292, 2): 445.44384827847034, (293, 2): 446.4242167379441, (294, 2): 448.37977106725066, (295, 2): 448.7531094289175, (296, 2): 449.82059512290243, (297, 2): 451.7577905291381, (298, 2): 452.5045159656825, (299,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           2): 454.4002907548327, (300, 2): 455.3592293970989, (301, 2): 456.0518160038497, (302, 2): 457.6815945195579, (303, 2): 458.80165555653946, (304, 2): 459.84489263118985, (305, 2): 461.7670243071667, (306, 2): 462.6930423727423, (307, 2): 463.2507916930532, (308, 2): 465.0688293909608, (309, 2): 466.0029807785587, (310, 2): 466.9272906826105, (311, 2): 468.6654084205942, (312, 2): 469.7430129867621, (313, 2): 471.6539231886929, (314, 2): 472.44264949357085, (315, 2): 473.52343541507264, (316, 2): 475.25125371447075, (317, 2): 476.2825422180785, (318, 2): 477.0656458001568, (319, 2): 478.45395317504926, (320, 2): 479.5823236747673, (321, 2):
                480.4182521125671, (322, 2): 482.1422903199968, (323, 2): 483.2764268459671, (324, 2): 483.8877775717456, (325, 2): 485.06915799006543, (326, 2): 486.2934294771169, (327, 2): 487.68534046984274, (328, 2): 489.12989758084115, (329, 2): 490.05808962589975, (330, 2): 491.2789538611026, (331, 2): 492.1659298681196, (332, 2): 492.9943273040805, (333, 2): 494.4936110003887, (334, 2): 496.3444985598132, (335, 2): 497.18313464894527, (336, 2): 498.38347812766847, (337, 2): 499.33429907735007, (338, 2): 499.91348413405234, (339, 2): 505.17465449392876, (222, 1): 349.54464621005843, (223, 1): 347.9679002187241, (224, 1): 345.37562417862625, (225, 1): 344.6121064275154, (226, 1): 341.84054420697635, (227, 1): 340.9771266838099, (228, 1): 340.20689525397194, (222, 2): 370.2537870860968, (223, 2): 372.4988751283046, (224, 2): 373.1687649356651, (225, 2): 373.8276781589581, (226, 2): 375.21921726899416, (227, 2): 375.6739936985021, (228, 2): 376.2749813596007}
    {(229, 1): 337.5798109501574, (230, 1): 336.49206225914395, (231, 1): 335.73645106011367, (232, 1): 333.0076909484552, (233, 1): 331.9797103258865, (234, 1): 331.35697480560276, (235, 1): 329.6865608801803, (236, 1): 327.70374774041227, (237, 1): 327.23057392348875, (238, 1): 324.90878844993034, (239, 1): 323.09568187504226, (240, 1): 322.3112425069401, (241, 1): 319.16569156895764, (242, 1): 318.2441375412664, (243, 1): 317.9112021202702, (244, 1): 315.1270169608514, (245, 1): 312.8043856360157, (246, 1): 312.82632990723283, (247, 1): 309.04794807752916, (248, 1): 308.2552594928032, (249, 1): 308.32117106001033, (250, 1): 305.13967619908243, (251, 1): 303.36086339862703, (252, 1): 303.478102870245, (253, 1): 299.3879045727717, (254, 1): 299.88133831573805, (255, 1): 295.9939868284946, (256, 1): 295.14401279165895, (257, 1): 295.4531664202027, (258, 1): 290.7997740084638, (259, 1): 289.69430332562365, (260, 1): 289.7753442133881, (261, 1): 285.9039223654013, (262, 1): 284.18460243037975, (263, 1): 284.5009034994792, (264, 1): 280.12654669777487, (265, 1): 278.45699387191297, (266, 1): 277.17855645844793, (267, 1): 274.27028859444505, (268, 1): 274.179869995939, (269, 1): 270.526982698317, (270, 1): 269.4080378324659, (271, 1): 268.7710498238442, (272, 1): 265.60750214668377,
     (273, 1): 264.35697223614324, (274, 1): 263.80424993373543, (275, 1): 259.77896323031274, (276, 1): 258.66925068789817, (277, 1): 257.88419168594226, (278, 1): 254.9950924143875, (279, 1): 253.77573766332816, (280, 1): 253.54695133503986, (281, 1): 249.73029310220426, (282, 1): 248.66078716980223, (283, 1): 245.46649452932107, (284, 1): 244.18748650469354, (285, 1): 243.6383635391516, (286, 1): 241.0915509633656, (287, 1): 240.0905409757874, (288, 1): 238.22896305786463, (289, 1):
     235.19027428995545, (290, 1): 234.15598431388028, (291, 1): 232.94272158856725, (292, 1): 230.3329664472858, (293, 1): 229.13663550829355, (294, 1): 227.71581678296852, (295, 1): 225.45245979865268, (296, 1): 224.0647719646442, (297, 1): 222.71810889553055, (298, 1): 219.78135030652544, (299, 1): 218.37383403831586, (300,
                                                                                                                                                                                                                                                                                                                                    1): 215.9945259055924, (301, 1): 214.97374626959876, (302, 1): 213.76158126831228, (303, 1): 211.27753925046744, (304, 1): 210.18158771699427, (305, 1): 208.16919748566414, (306, 1): 205.82158273754044, (307, 1): 204.69892671744304, (308, 1): 203.71497084640993, (309, 1): 200.82856845598567, (310, 1): 199.33180102161162, (311, 1): 198.054334837862, (312, 1): 195.16715412906063, (313, 1): 193.63607825502135, (314, 1): 191.84322994708722, (315, 1): 190.1867467270778, (316, 1): 188.39738415038988, (317, 1): 186.02131069886627, (318, 1): 184.9574028666912, (319, 1): 183.5810101369358, (320, 1): 181.11888829255332, (321, 1): 179.0051536292958, (322, 1): 178.19749744286122, (323, 1): 176.16121199776018, (324, 1): 174.83619033605478, (325, 1): 173.99962900667776, (326, 1): 171.38556502229437, (327, 1): 169.70910947426373, (328, 1): 167.47417585327244, (329, 1): 166.0320419223806, (330, 1): 164.43566423514528, (331, 1): 162.46939806181962, (332, 1): 160.94105385603845, (333, 1): 159.02400942328092, (334, 1): 156.56030812780898, (335, 1): 155.3470890264324, (336, 1): 154.1786009277337, (337, 1): 151.94395611246125, (338, 1):
     149.2429127629363, (339, 1): 151.88303732245, (229, 2): 378.7824350704452, (230, 2): 378.31898161740173, (231, 2): 379.1799605750242, (232, 2): 380.71016928465485, (233, 2): 380.82352073207505, (234, 2): 382.0974395916424, (235, 2): 385.6332180772579, (236, 2): 384.8926179711728, (237, 2): 387.6370254172557, (238, 2): 387.13511281949565, (239, 2): 387.444097521273, (240, 2): 390.6877073597684, (241, 2): 390.437094174019, (242, 2): 390.6012891325465, (243, 2): 393.3973697009334, (244, 2): 393.232028109547, (245, 2): 393.61283680744447, (246, 2): 396.96848349602675, (247, 2): 396.7893125661146, (248, 2): 397.0096494159833, (249, 2): 399.8627247603104, (250, 2): 399.7450113049305, (251, 2): 400.0667994433941, (252, 2): 403.21615568942985, (253, 2): 403.0654142517175, (254, 2): 405.7172861068198, (255,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          2): 404.8076106555048, (256, 2): 406.0293532124265, (257, 2): 408.81203277755134, (258, 2): 408.05373297683127, (259, 2): 409.06964514456297, (260, 2): 411.9617390145371, (261, 2): 411.20672261493064, (262, 2): 412.2358897860989, (263, 2): 414.7001851104991, (264, 2): 414.0993599769197, (265, 2): 415.0828676866149, (266, 2): 417.42990202889615, (267, 2): 417.4916436041819, (268, 2): 419.8619489372302, (269, 2): 419.93669878870156, (270, 2): 420.4561658607602, (271, 2): 422.5076439325268, (272, 2): 423.3027261194376, (273, 2): 423.76213253224216, (274, 2): 426.01535333984793, (275, 2): 426.2535828579153, (276, 2): 427.08008562204435, (277, 2): 429.3455663939063, (278, 2): 429.70256532714666, (279, 2): 430.44672535974746, (280, 2): 432.4152045868102, (281, 2): 432.73648789394, (282, 2): 435.18165750709466, (283, 2): 435.3383382494676, (284, 2): 436.07524231453505, (285, 2): 438.267791786281, (286, 2): 438.59595451238476, (287, 2): 439.4005273853498, (288, 2):
     441.64345423644136, (289, 2): 442.0602391621859, (290, 2): 442.8683754388892, (291, 2): 445.023121158447, (292, 2): 445.44384827847034, (293, 2): 446.4242167379441, (294, 2): 448.37977106725066, (295, 2): 448.7531094289175, (296, 2): 449.82059512290243, (297, 2): 451.7577905291381, (298, 2): 452.5045159656825, (299, 2): 454.4002907548327, (300, 2): 455.3592293970989, (301, 2): 456.0518160038497, (302, 2): 457.6815945195579, (303, 2): 458.80165555653946, (304, 2): 459.84489263118985, (305, 2): 461.7670243071667, (306, 2): 462.6930423727423, (307, 2): 463.2507916930532, (308, 2): 465.0688293909608, (309, 2): 466.0029807785587, (310, 2): 466.9272906826105, (311, 2): 468.6654084205942, (312, 2): 469.7430129867621, (313, 2): 471.6539231886929, (314, 2): 472.44264949357085, (315, 2): 473.52343541507264, (316, 2): 475.25125371447075, (317, 2): 476.2825422180785, (318, 2): 477.0656458001568, (319, 2): 478.45395317504926, (320, 2): 479.5823236747673, (321, 2): 480.4182521125671, (322, 2): 482.1422903199968, (323, 2): 483.2764268459671, (324, 2): 483.8877775717456, (325, 2): 485.06915799006543, (326, 2): 486.2934294771169, (327, 2): 487.68534046984274, (328, 2): 489.12989758084115, (329, 2): 490.05808962589975, (330, 2): 491.2789538611026, (331, 2): 492.1659298681196, (332, 2): 492.9943273040805, (333, 2): 494.4936110003887, (334, 2): 496.3444985598132, (335, 2): 497.18313464894527, (336, 2): 498.38347812766847, (337, 2): 499.33429907735007, (338, 2): 499.91348413405234, (339, 2): 505.17465449392876, (222, 1): 349.41375328585724, (223, 1): 347.62026482380816, (224, 1): 345.2815494611527, (225, 1): 344.46906701298576, (226, 1): 341.58199569358743, (227, 1): 340.81963246713104, (228, 1): 340.07990220191465, (222, 2): 370.2537870860968, (223, 2): 372.4988751283046, (224, 2): 373.1687649356651, (225, 2): 373.8276781589581, (226, 2): 375.21921726899416, (227, 2): 375.6739936985021, (228, 2): 376.2749813596007}
    {(229, 1): 337.5798109501574, (230, 1): 336.49206225914395, (231, 1): 335.73645106011367, (232, 1): 333.0076909484552, (233, 1): 331.9797103258865, (234, 1): 331.35697480560276, (235, 1): 329.6865608801803, (236, 1): 327.70374774041227, (237, 1): 327.23057392348875, (238, 1): 324.90878844993034, (239, 1): 323.09568187504226, (240, 1): 322.3112425069401, (241, 1): 319.16569156895764, (242, 1): 318.2441375412664, (243, 1): 317.9112021202702, (244, 1): 315.1270169608514, (245, 1): 312.8043856360157, (246, 1): 312.82632990723283, (247, 1): 309.04794807752916, (248, 1): 308.2552594928032, (249, 1): 308.32117106001033, (250, 1): 305.13967619908243, (251, 1): 303.36086339862703, (252, 1): 303.478102870245, (253, 1): 299.3879045727717, (254, 1): 299.88133831573805, (255, 1): 295.9939868284946, (256, 1): 295.14401279165895, (257, 1): 295.4531664202027, (258, 1): 290.7997740084638, (259, 1): 289.69430332562365, (260, 1): 289.7753442133881, (261, 1): 285.9039223654013, (262, 1): 284.18460243037975, (263, 1): 284.5009034994792, (264, 1): 280.12654669777487, (265, 1): 278.45699387191297, (266, 1): 277.17855645844793, (267, 1): 274.27028859444505, (268, 1): 274.179869995939, (269, 1): 270.526982698317, (270, 1): 269.4080378324659, (271, 1): 268.7710498238442, (272, 1): 265.60750214668377,
     (273, 1): 264.35697223614324, (274, 1): 263.80424993373543, (275, 1): 259.77896323031274, (276, 1): 258.66925068789817, (277, 1): 257.88419168594226, (278, 1): 254.9950924143875, (279, 1): 253.77573766332816, (280, 1): 253.54695133503986, (281, 1): 249.73029310220426, (282, 1): 248.66078716980223, (283, 1): 245.46649452932107, (284, 1): 244.18748650469354, (285, 1): 243.6383635391516, (286, 1): 241.0915509633656, (287, 1): 240.0905409757874, (288, 1): 238.22896305786463, (289, 1):
     235.19027428995545, (290, 1): 234.15598431388028, (291, 1): 232.94272158856725, (292, 1): 230.3329664472858, (293, 1): 229.13663550829355, (294, 1): 227.71581678296852, (295, 1): 225.45245979865268, (296, 1): 224.0647719646442, (297, 1): 222.71810889553055, (298, 1): 219.78135030652544, (299, 1): 218.37383403831586, (300,
                                                                                                                                                                                                                                                                                                                                    1): 215.9945259055924, (301, 1): 214.97374626959876, (302, 1): 213.76158126831228, (303, 1): 211.27753925046744, (304, 1): 210.18158771699427, (305, 1): 208.16919748566414, (306, 1): 205.82158273754044, (307, 1): 204.69892671744304, (308, 1): 203.71497084640993, (309, 1): 200.82856845598567, (310, 1): 199.33180102161162, (311, 1): 198.054334837862, (312, 1): 195.16715412906063, (313, 1): 193.63607825502135, (314, 1): 191.84322994708722, (315, 1): 190.1867467270778, (316, 1): 188.39738415038988, (317, 1): 186.02131069886627, (318, 1): 184.9574028666912, (319, 1): 183.5810101369358, (320, 1): 181.11888829255332, (321, 1): 179.0051536292958, (322, 1): 178.19749744286122, (323, 1): 176.16121199776018, (324, 1): 174.83619033605478, (325, 1): 173.99962900667776, (326, 1): 171.38556502229437, (327, 1): 169.70910947426373, (328, 1): 167.47417585327244, (329, 1): 166.0320419223806, (330, 1): 164.43566423514528, (331, 1): 162.46939806181962, (332, 1): 160.94105385603845, (333, 1): 159.02400942328092, (334, 1): 156.56030812780898, (335, 1): 155.3470890264324, (336, 1): 154.1786009277337, (337, 1): 151.94395611246125, (338, 1):
     149.2429127629363, (339, 1): 151.88303732245, (229, 2): 378.7279658362461, (230, 2): 378.34594994535263, (231, 2): 379.29163413625326, (232, 2): 380.9291012163756, (233, 2): 381.2265093110849, (234, 2): 382.44998098381336, (235, 2): 386.1290512327991, (236, 2): 385.13447029951436, (237, 2): 388.2482266819874, (238, 2): 387.48616235184363, (239, 2): 387.74331125943064, (240, 2): 391.1001104120481, (241, 2): 390.77618524472786, (242, 2): 391.0130780828977, (243, 2): 394.0271758798897, (244, 2): 393.57923979640896, (245, 2): 393.90536846986964, (246, 2): 397.5539530602206, (247, 2): 397.1144557884964, (248, 2): 397.40526729911033, (249, 2): 400.45640456975394, (250, 2): 400.0719088572552, (251, 2): 400.42055418543566, (252, 2): 403.74436616553294, (253, 2): 403.45882384734955, (254, 2): 406.36028842654093, (255, 2): 405.1716292735468, (256, 2): 406.3845908252144, (257, 2): 409.3700478988897, (258, 2): 408.4080051509487, (259, 2): 409.46288719683827, (260, 2): 412.62536400523885, (261, 2): 411.5549363993794, (262, 2): 412.6621485505694, (263, 2): 415.3460831720305, (264, 2): 414.4914253218336, (265, 2): 415.475586126225, (266, 2): 417.97955082337324, (267, 2): 417.82857451745633, (268, 2): 420.5317526552469, (269, 2): 420.33520551666896, (270, 2): 420.79450169267415, (271, 2): 423.16848931997987, (272, 2): 423.64713816510783, (273, 2): 424.0883520839654, (274, 2): 426.8665435118254, (275, 2): 426.67914097674185, (276, 2): 427.5125142314114, (277, 2): 430.01742599890525, (278, 2): 430.18954374019165, (279, 2): 430.9834350355015, (280, 2): 433.3541535707597, (281, 2): 433.22212297334, (282, 2): 435.98504166056904, (283, 2): 435.8397400547887, (284, 2): 436.5474627188795, (285, 2): 439.1785846971751, (286, 2): 439.08715471369436, (287, 2): 439.85986650088995, (288, 2): 442.4474519401106, (289, 2): 442.5330536913633, (290, 2): 443.3091877715092, (291, 2): 445.79764462037383, (292, 2): 445.9014718866429, (293, 2): 446.84301775512154, (294, 2): 449.2058071405212, (295, 2): 449.19848629553235, (296, 2): 450.2632995560355, (297, 2): 452.55448025602647, (298, 2): 452.9201371334415,
     (299, 2): 455.1455338470001, (300, 2): 455.7806039129738, (301, 2): 456.4457640900972, (302, 2): 458.36696764287075, (303, 2): 459.2055099965005, (304, 2): 460.20743488299473, (305, 2): 462.5218290388766, (306, 2): 463.1011978947817, (307, 2): 463.6368603420083, (308, 2): 465.99349995869443, (309, 2): 466.51766273972777, (310, 2): 467.32616997157913, (311, 2): 469.62453362055203, (312, 2): 470.2283896724863, (313, 2): 472.534740008336, (314, 2): 472.93992057362544, (315, 2): 473.97790737263574, (316, 2): 476.16489866630957, (317, 2): 476.7465535462426, (318, 2): 477.5382462264706, (319, 2): 479.27266219479634, (320, 2): 480.0344636572465, (321, 2): 480.87688932890944, (322, 2): 482.9867624606412, (323, 2): 483.74068034286677, (324, 2): 484.3278215799045, (325, 2): 485.919307528046, (326, 2): 486.71740966982304, (327, 2): 488.5652864681424, (328, 2): 489.56035363663983, (329, 2): 490.4517892767141, (330, 2): 492.1492802395901, (331, 2): 492.63455904534203, (332, 2): 493.39055194283895, (333, 2): 495.3108094701842, (334, 2): 496.76645641684456, (335, 2): 497.53228251684703, (336, 2): 499.48015377501775, (337, 2): 499.875460833887, (338, 2): 500.47131016234385, (339, 2): 505.3511614040502, (222, 1): 349.41375328585724, (223, 1): 347.62026482380816, (224, 1): 345.2815494611527, (225, 1): 344.46906701298576, (226, 1): 341.58199569358743, (227, 1): 340.81963246713104, (228, 1): 340.07990220191465, (222, 2): 370.16871134437486, (223, 2): 371.99435549416404, (224, 2): 373.00368184421274, (225, 2): 373.6365021038634, (226, 2): 374.89167166904235, (227, 2): 375.52850354804144, (228, 2): 376.10569209776054}


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
