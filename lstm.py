# This code is junk i know sorry

import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1209809284)

# Get all labels
AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'NOLABEL']
labels2Tensor = {}
for label in AllPossibleLabels:
  labels2Tensor[label] = torch.tensor([len(labels2Tensor)])

NUMLABELS = len(AllPossibleLabels)
PREDICT_EVERY_NTH_FRAME = 15
PRECOMPUTE = False
WINDOWWIDTH = 30*18
WINDOWSTEP = WINDOWWIDTH // 2

# 0 : rightTO
# 1 : lcRel
# 2 : cutin
# 3 : cutout
# 4 : evtEnd
# 5 : objTurnOff
# 6 : end
# 7 : NOLABEL

# It would be nice if we could increase the window of the label predictions
# A problem is that having to choose the correct label on the precise frame number
# is too demanding of the network. The exact frame number is noisy and is a human
# choice. But the frame number plus or minus 15 frames is less subject to human bias.

# Use attention? bidirectional lstm?
# how many lstm layers to use? dont wanna overfit


def clamp(l, h, x):
  if x < l: return l
  if x > h: return h
  return x

class mylstm(nn.Module):
  def __init__(self, hidden_dim, input_dim, output_size):
    super(mylstm, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1)
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, bidirectional=True)
    self.hidden2out = nn.Linear(2*hidden_dim, output_size)

  def forward(self, data):
    # see nn.LSTM documentation for input and output shapes
    context_seq = self.encoder(data)  # == h_T, (hs, cs)
    context_seq = context_seq[1]  # == (hs,cs)
    context_seq = context_seq[0]  # == hs
    context_seq = context_seq[-1]  # == hs for last layer, i think
    context_seq = context_seq.view(context_seq.shape[0], 1, context_seq.shape[1])
    lstm_out, _ = self.lstm(context_seq)
    lstm_out = lstm_out[::PREDICT_EVERY_NTH_FRAME]
    out_space = self.hidden2out(lstm_out.view(lstm_out.shape[0], -1))
    out_scores = F.log_softmax(out_space, dim=1)
    return out_scores


def evaluate(model, loss_function, sequences, saveFileName):
  model.eval()
  outputs = []
  average_loss = 0
  for x, y in sequences:
    model.zero_grad()
    yhat = model(x)
    yhat = yhat.argmax(dim=1)
    loss = float(loss_function(model(x), y))
    average_loss += loss
    yhat = yhat.cpu().numpy().tolist()
    y = y.cpu().numpy().tolist()
    yhat = ''.join(['_' if z == 7 else str(z) for z in yhat]) + '  ' + str(loss) + '\n'
    y = ''.join(['.' if z == 7 else str(z) for z in y]) + '\n\n'
    outputs.extend([yhat, y])
  with open(saveFileName, 'w') as f:
    f.writelines(outputs)
    f.write('average loss =' + str(average_loss/len(sequences)) + '\n')
  model.train()


def checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences):
  with torch.no_grad():
    print('Saving model at epoch:', epoch)
    torch.save(model.state_dict(), 'model.pt')
    with open('losses.pkl', 'wb') as file:
      pickle.dump(losses, file)
    evaluate(model, loss_function, trainSequences, saveFileName='trainOutputs.txt')
    evaluate(model, loss_function, testSequences, saveFileName='testOutputs.txt')


def train(trainSequences, testSequences):
  print('Training')
  model = mylstm(64, 17, NUMLABELS)
  model.to(torch.device('cuda'))
  model.train()

  class_weights = [1, # rightTO
                   1, # lcRel
                   1, # cutin
                   1, # cutout
                   1, # evtEnd
                   1, # objTurnOff
                   1, # end
                   1/10]# NOLABEL
  class_weights = torch.tensor(class_weights, device=torch.device('cuda'))/sum(class_weights)

  # loss_function = nn.CrossEntropyLoss(weight=class_weight)
  loss_function = nn.NLLLoss(weight=class_weights)
  # optimizer = optim.SGD(model.parameters(), lr=0.1)
  optimizer = optim.Adam(model.parameters())

  print('Enter training loop')
  ema_loss = None
  losses = []
  for epoch in range(5000):
    for i, (x, y) in enumerate(trainSequences):
      loss = loss_function(model(x), y)
      losses.append(float(loss))
      loss.backward()
      if i % 10 == 0:  # use minibatches of size 10
        optimizer.step()
        model.zero_grad()

    if epoch % 100 == 0 and epoch > 0:
      checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences)
    if ema_loss is None:
      ema_loss = losses[-1]
    ema_loss = .6 * ema_loss + .4 * losses[-1]
    print(epoch, '/', 5000, '  ', ema_loss)

  print('Finished training')
  checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences)


def loadFeaturesAndLabels(files):
  # returns a list of (path, [label, framnum], features)
  # where features == (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs)
  ret = []
  for file in files:
    labelsFilePath = file.replace('featuresLSTM/', 'groundTruthLabels/').replace('_m0.pkl', '_labels.txt')
    if not os.path.isfile(labelsFilePath):
      continue
    videodata = []

    # Get features
    with open(file, 'rb') as featuresfile:
      videodata.append(list(pickle.load(featuresfile)))

    # Get labels
    with open(labelsFilePath) as labelsFile:
      labels = []
      lines = labelsFile.readlines()
      labelLines = [line.rstrip('\n') for line in lines]
      for line in labelLines:
        label, labelTime = line.split(',')
        label = label.split('=')[0]
        if label == 'barrier':
          continue
        frameNumber = int((float(labelTime) % 300) * 30)
        labels.append((label, frameNumber))
      videodata.append(labels)

    ret.append(videodata)
  return ret


def getSequenceForInterval(low, high, features, labels, frameNum2LabelTensors):
  xs, ys = [], []

  (_, lanescores, vehicles, boxcornerprobs) = features
  # Only look at data from this interval of frames
  vehicles = vehicles[low:high]
  probs = boxcornerprobs[low:high]
  lanescores = lanescores[low:high]

  # for each frame in the interval
  for j, (vehicles, probs, lanescores_) in enumerate(zip(vehicles, probs, lanescores)):
    lanescores_ = [float(score.cpu().numpy()) for score in lanescores_]

    # for some reason I organized the pkl file s.t. we have to do this
    probsleft = probs[:len(vehicles)]  # left lane line probability map values at box corners for each vehicle
    probsright = probs[len(vehicles):]  # See LaneLineDetectorERFNet.py::175

    features_xs = []
    # Create tensors
    for vehicle, probleft, probright in zip(vehicles, probsleft, probsright):
      # Put object id into sensible range
      objectid, x1, y1, x2, y2 = vehicle
      objectid = (objectid % 1000) / 1000
      vehicle = (objectid, x1 / 720, y1 / 480, x2 / 720, y2 / 480)

      features_xs.append(torch.tensor([[*vehicle, *probleft, *probright, *lanescores_]], dtype=torch.float))
    if len(features_xs) == 0:
      features_xs = [torch.tensor([[0]*13 + [*lanescores_]], dtype=torch.float)]  # make sure there is always an input tensor
    features_xs = torch.cat(features_xs)

    xs.append(features_xs)
    if (low+j) % PREDICT_EVERY_NTH_FRAME == 0:
      if (low+j)//PREDICT_EVERY_NTH_FRAME in frameNum2LabelTensors:
        ys.append(frameNum2LabelTensors[(low+j)//PREDICT_EVERY_NTH_FRAME])
      else:
        ys.append(labels2Tensor['NOLABEL'])

  xs = pad_sequence(xs).to(torch.device('cuda'))
  xs = torch.flip(xs, dims=(0,))
  ys = torch.cat(ys).to(torch.device('cuda'))
  return (xs, ys)


def getSequencesForData(data):
  # Make input tensors from the data
  sequences = []
  for features, labels in data:
    frameNum2LabelTensors = {}
    for label, frameNum in labels:
      while frameNum//PREDICT_EVERY_NTH_FRAME in frameNum2LabelTensors:
        frameNum += PREDICT_EVERY_NTH_FRAME
      frameNum2LabelTensors[frameNum//PREDICT_EVERY_NTH_FRAME] = labels2Tensor[label]
    for i in range(0, 30*60*5 - WINDOWWIDTH, WINDOWSTEP):
      low = i
      high = i + WINDOWWIDTH
      # Search for a label that is in the interval.
      # If such a label does not exist, then we will not add the sequence to the dataset.
      for label, frameNum in labels:
        if low <= frameNum <= high:
          (xs, ys) = getSequenceForInterval(low, high, features, labels, frameNum2LabelTensors)
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

    trainSize = int(len(filepaths)*.8)
    trainSet = filepaths[:trainSize]
    testSet = filepaths[trainSize:]

    print('Train set has size:', len(trainSet))
    for path in trainSet: print(path)
    print('Test set has size:', len(testSet))
    for path in testSet: print(path)

    # Collect labels and features into one place
    trainData = loadFeaturesAndLabels(trainSet)
    testData = loadFeaturesAndLabels(testSet)

    # Convert the data into tensors
    trainSequences = getSequencesForData(trainData)
    testSequences = getSequencesForData(testData)

    with open('tensors.pkl', 'wb') as file:
      pickle.dump((trainSequences, testSequences), file)
    print('Wrote tensors pickle. Exiting.')
  else:
    with open('tensors.pkl', 'rb') as file:
      (trainSequences, testSequences) = pickle.load(file)
    train(trainSequences, testSequences)
