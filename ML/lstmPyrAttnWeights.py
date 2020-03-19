import os, time, pickle, random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1209809284)

device = torch.device('cuda')
PRECOMPUTE = False

# Get all labels
AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'NOLABEL']
labels2Tensor = {}
for label in AllPossibleLabels:
  labels2Tensor[label] = torch.tensor([len(labels2Tensor)])

NUMLABELS = len(AllPossibleLabels)
PREDICT_EVERY_NTH_FRAME = 8
WINDOWWIDTH = 30*8
WINDOWSTEP = WINDOWWIDTH // 4

# 0 : rightTO
# 1 : lcRel
# 2 : cutin
# 3 : cutout
# 4 : evtEnd
# 5 : objTurnOff
# 6 : end
# 7 : NOLABEL

def clamp(l, h, x):
  if x < l: return l
  if x > h: return h
  return x

def one_hot(i):
    return torch.tensor([[0. if j != i else 1. for j in range(NUMLABELS)]], device=device)

class mylstm(nn.Module):
  def __init__(self, hidden_dim, input_dim, output_dim):
    super(mylstm, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.encoder = nn.LSTM(input_dim, hidden_dim)
    self.pencoder1 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True)
    self.pencoder2 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True)
    self.attn = nn.Linear(hidden_dim + output_dim, WINDOWWIDTH//4)
    self.attn_combine = nn.Linear(hidden_dim*2 + output_dim, hidden_dim)
    self.gru = nn.GRU(hidden_dim, hidden_dim)
    self.out = nn.Linear(hidden_dim, output_dim)

  def forward(self, data:torch.Tensor, y:torch.Tensor=None):
    _, context_seq = self.encoder(data)
    context_seq = context_seq[0].view(data.shape[1]//2,1,-1)
    context_seq, _ = self.pencoder1(context_seq)
    context_seq = context_seq.view(data.shape[1]//4,1,-1)
    context_seq, hidden = self.pencoder2(context_seq)
    context_seq = context_seq.view(1,WINDOWWIDTH//4,self.hidden_dim*2)

    hidden = hidden[0][0].view(1,1,-1)
    input = one_hot(NUMLABELS-1)
    outputs = torch.zeros((WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME, NUMLABELS), device=device)
    for i in range(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME):
      attn_weights = F.softmax(self.attn(torch.cat((input, hidden[0]), 1)), dim=1)
      attn_applied = torch.bmm(attn_weights.unsqueeze(0), context_seq)
      output = torch.cat((input, attn_applied[0]), 1)
      output = self.attn_combine(output).unsqueeze(0)
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
      outputs[i] += F.log_softmax(self.out(output[0]), dim=1)[0]
      if y is not None:
        input = one_hot(y[i])
      else:
        input = outputs[i].view(1,NUMLABELS)
    return outputs

def evaluate(model, lossFunction, sequences, saveFileName):
  outputs = []
  avgloss = 0
  for x, y in sequences:
    yhat = model(x)
    avgloss += float(lossFunction(yhat, y))
    yhat = yhat.argmax(dim=1)
    yhat = yhat.cpu().numpy().tolist()
    y = y.cpu().numpy().tolist()
    yhat = ''.join(['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in yhat]) + ' '
    y = ''.join(['.' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in y]) + '\n\n'
    outputs.extend([yhat, y])
  with open(saveFileName, 'w') as f:
    f.writelines(outputs)
  return avgloss / len(sequences)

def checkpoint(epoch, trainloss, testloss, model, lossFunction, trainSequences, testSequences):
  model.eval()
  with torch.no_grad():
    avgtrainloss = evaluate(model, lossFunction, trainSequences, 'trainOutputs.txt')
    avgtestloss = evaluate(model, lossFunction, testSequences, 'testOutputs.txt')
    if len(trainloss) and avgtrainloss < min(trainloss):
      mintrainloss = avgtrainloss
      torch.save(model.state_dict(), 'mintrainlossmodel.pt')
    if len(testloss) and avgtestloss < min(testloss):
      mintestloss = avgtestloss
      torch.save(model.state_dict(), 'mintestlossmodel.pt')
    trainloss.append(avgtrainloss)
    testloss.append(avgtestloss)
    torch.save(model.state_dict(), 'mostrecentmodel.pt')
    with open('trainloss.pkl', 'wb') as file:
      pickle.dump(trainloss, file)
    with open('testloss.pkl', 'wb') as file:
      pickle.dump(testloss, file)
  model.train()

def train(trainSequences, testSequences):
  print('Training')
  model = mylstm(128, 17, NUMLABELS)
  model.to(device)
  model.train()

  # class_counts = {'rightTO': 2084,
  #                 'lcRel': 1056,
  #                 'cutin': 520,
  #                 'cutout': 788,
  #                 'evtEnd': 2352,
  #                 'objTurnOff': 48,
  #                 'end': 20,
  #                 'NOLABEL': 110732}
  class_counts = {'rightTO': 1,
                  'lcRel': 1,
                  'cutin': 1,
                  'cutout': 1,
                  'evtEnd': 1,
                  'objTurnOff': 1,
                  'end': 1,
                  'NOLABEL': -2 + WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME}
  class_weights = [1/class_counts[lab] for lab in AllPossibleLabels]
  class_weights = torch.tensor(class_weights, device=device) / sum(class_weights)

  lossFunction = nn.NLLLoss(weight=class_weights)
  optimizer = optim.Adam(model.parameters())

  print('Enter training loop')
  trainloss = []
  testloss = []
  for epoch in range(5000):
    start = time.time()
    loss = 0
    for i in range(len(trainSequences)):
      k = random.randint(0,len(trainSequences)-1)
      x,y = trainSequences[k]
      if k%10: loss += lossFunction(model(x,y), y)
      else:    loss += lossFunction(model(x), y)
      if i % 10 == 9: # use a minibatch of size 10
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = 0

    if epoch % 3 == 0:
      checkpoint(epoch, trainloss, testloss, model, lossFunction, trainSequences, testSequences)

    end = time.time()
    print('epoch {}    train {:1.5}    test {:1.5}    time {}'.format(epoch, trainloss[-1], testloss[-1], int(end-start)))

  print('Finished training')
  checkpoint(epoch, trainloss, testloss, model, lossFunction, trainSequences, testSequences)

# gxs = []
mean = torch.tensor([4.7812e-01, 3.8052e-01, 4.2687e-01, 4.4340e-01, 4.9411e-01, 3.1371e-04,
        6.1817e-02, 7.6253e-02, 3.6926e-04, 1.6677e-04, 6.1989e-02, 4.9159e-02,
        3.5818e-04, 4.1619e-01, 9.3100e-01, 9.2963e-01, 4.6476e-01])
std = torch.tensor([.307,.2182,.0546,.2191,.0504,.004,0.1959,.2156,.0038,0.0023,.1858,.1734,.0037,.4054,.184,0.1859,.3852])
class_counts = {label:0 for label in AllPossibleLabels}

def getSequenceForInterval(low, high, features, labels, frameNum2Label, noise):
  xs, ys = [], []

  (_, lanescores, vehicles, boxcentroidprobs) = features

  # Only look at data from this interval of frames
  vehicles = vehicles[low:high]
  probs = boxcentroidprobs[low:high]
  lanescores = lanescores[low:high]
  # for the jth frame in the interval
  for j, (vehicles_, probs_, lanescores_) in enumerate(zip(vehicles, probs, lanescores)):
    lanescores_ = [float(score.cpu().numpy()) for score in lanescores_]
    probsleft = probs_[:len(vehicles_)]  # left lane line probability map values at box corners for each vehicle
    probsright = probs_[len(vehicles_):]  # See LaneLineDetectorERFNet.py::175

    features_xs = []
    # sort by objectid
    stuff = sorted(list(zip(vehicles_, probsleft, probsright)), key=lambda x:-x[0][0])
    for vehicle, probleft, probright in stuff:
      # Put object id into sensible range
      objectid, x1, y1, x2, y2 = vehicle
      objectid = (objectid % 1000) / 1000
      vehicle = (objectid, x1 / 720, y1 / 480, x2 / 720, y2 / 480)
      tens = torch.tensor([[*vehicle, *probleft, *probright, *lanescores_]])
      tens = (tens - mean) / std
      if noise:
        tens += torch.randn(tens.size()) * .05
      features_xs.append(tens)
    # gxs.extend(features_xs)
    if len(features_xs) == 0:
      features_xs = [torch.tensor([[0]*13 + [*lanescores_]])]  # make sure there is always an input tensor
      # ugly, should standardize lanescores too even on zero inputs
      features_xs[0][0][13:] = (features_xs[0][0][13:] - mean[13:]) / std[13:]
    features_xs = torch.cat(features_xs)

    xs.append(features_xs)
    if (low+j) % PREDICT_EVERY_NTH_FRAME == 0:
      if (low+j)//PREDICT_EVERY_NTH_FRAME in frameNum2Label:
        label = frameNum2Label[(low+j)//PREDICT_EVERY_NTH_FRAME]
        class_counts[label] += 1
        ys.append(labels2Tensor[label])
      else:
        class_counts['NOLABEL'] += 1
        ys.append(labels2Tensor['NOLABEL'])

  xs = pad_sequence(xs).to(device)
  xs = torch.flip(xs, dims=(0,))
  ys = torch.cat(ys).to(device)
  return (xs,ys)

def getSequencesForFiles(files):
  sequences = []
  for file in files:
    labelsFilePath = file.replace('featuresLSTM/', 'groundTruthLabels/').replace('_m0.pkl', '_labels.txt')
    if not os.path.isfile(labelsFilePath):
      continue

    # Get features
    features = None
    with open(file, 'rb') as featuresfile:
      features = list(pickle.load(featuresfile))

    # Get labels
    labels = []
    with open(labelsFilePath) as labelsFile:
      lines = labelsFile.readlines()
      labelLines = [line.rstrip('\n') for line in lines]
      for line in labelLines:
        label, labelTime = line.split(',')
        label = label.split('=')[0]
        if label == 'barrier':
          continue
        frameNumber = int((float(labelTime) % 300) * 30)
        labels.append((label, frameNumber))

    # Make input tensors from the data
    # First do labels tensors
    frameNum2Label= {}
    for label, frameNum in labels:
      while frameNum//PREDICT_EVERY_NTH_FRAME in frameNum2Label:
        frameNum += PREDICT_EVERY_NTH_FRAME
      frameNum2Label[frameNum//PREDICT_EVERY_NTH_FRAME] = label

    # Then do features tensors
    for noise in [False, True]:
      for i in range(0, 30*60*5 - WINDOWWIDTH, WINDOWSTEP):
        [low, high] = [i, i + WINDOWWIDTH]
        # Search for a label that is in the interval.
        # If such a label does not exist, then we will not add the sequence to the dataset.
        for label, frameNum in labels:
          if low <= frameNum <= high:
            (xs, ys) = getSequenceForInterval(low, high, features, labels, frameNum2Label, noise)
            sequences.append((xs, ys))
            # There may be more than one label in this interval.
            # If we ran this loop twice in this interval then we would append the same exact (xs,ys) to sequences
            break
  return sequences

if __name__ == '__main__':
  print('Loading data.')

  if PRECOMPUTE:
    # Get paths to precomputed features
    filepaths = []
    for (dirpath, dirnames, filenames) in os.walk('precomputed/featuresLSTM'):
      filepaths.extend(dirpath + '/' + f for f in filenames)
    random.shuffle(filepaths)

    trainSize = int(len(filepaths)*.85)
    trainSet = filepaths[:trainSize]
    testSet = filepaths[trainSize:]

    print('Train set has size:', len(trainSet))
    for path in trainSet: print(path)
    print('Test set has size:', len(testSet))
    for path in testSet: print(path)

    # Convert the data into tensors
    trainSequences = getSequencesForFiles(trainSet)
    testSequences = getSequencesForFiles(testSet)

    # bigboy = torch.cat(gxs)
    # print(bigboy.mean(dim=0))
    # print(bigboy.std(dim=0))
    # print(bigboy.shape)
    # exit(1)

    print('Class counts for data:',class_counts)

    with open('tensors.pkl', 'wb') as file:
      pickle.dump((trainSequences, testSequences), file)
    print('Wrote tensors pickle. Exiting.')
  else:
    with open('tensors.pkl', 'rb') as file:
      (trainSequences, testSequences) = pickle.load(file)
    train(trainSequences, testSequences)
