import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
from erfnet.models.erfnet import ERFNet
import torch.nn.functional as F
import numpy as np

class LaneLineDetector:
    def __init__(self):
        num_class = 5
        self.model = ERFNet(num_class)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        state_dict = torch.load('erfnet/trained/ERFNet_trained.tar')['state_dict']

        #https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)


        cudnn.benchmark = True
        cudnn.fastest = True
        self.model.eval()

    def getLines(self, frame):
        scale = 976 / 720
        width = int(frame.shape[1] * scale + .5)
        height = int(frame.shape[0] * scale + .5)
        dsize = (width, height)
        frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_LINEAR)
        frame = frame[281:281+208,:976]
        frame_orig = frame
        with torch.no_grad():
            PERMUTATION = (2,0,1) # rotate right
            INV_PERMUTATION = (1,2,0) # rotate left

            frame = np.transpose(frame,PERMUTATION)
            input_mean = np.array(self.model.input_mean)[:,np.newaxis,np.newaxis]
            input_std = np.array(self.model.input_std)[:,np.newaxis,np.newaxis]
            frame = (frame - input_mean) / input_std
            frame = torch.from_numpy(frame.astype('float32')).to(self.device)
            frame = frame.unsqueeze(0) # add empty batch dimension

            input_var = torch.autograd.Variable(frame)
            output, lane_scores = self.model(input_var)
            output = F.softmax(output, dim=1)

            prob_map = output.data.cpu().numpy() # BxCxHxW = batch x channel x height x width
            # prob_of_lane = lane_scores.data.cpu().numpy() # BxO

            # OUTPUT FORMAT
            # prob_map[0] is the output for the ith image in the batch
            # prob_map[0,0,:,:] is the probability map for all 4 lane lines
            # prob_map[0,i,:,:] is the probability map for the ith lane line (1<=i<=4)
            #
            # prob_of_lane is a probability that the ith lane exists (1<=i<=4) (i.e. gives a score)

            prob_map = prob_map[:,0]
            makeRGB = (1.-np.transpose(np.vstack([prob_map]*3), INV_PERMUTATION))
            return ((makeRGB * 255 + frame_orig)*.5).astype('int8')