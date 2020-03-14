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
torch.manual_seed(1)
make_new = False
load_model = True
testing = True
# 0 : rightTO
# 1 : lcRel
# 2 : cutin
# 3 : cutout
# 4 : evtEnd
# 5 : objTurnOff
# 6 : end
# 7 : barrier # TODO consider removing this
# 8 : NOLABEL

# It would be nice if we could increase the window of the label predictions
# A problem is that having to choose the correct label on the precise frame number
# is too demanding of the network. The exact frame number is noisy and is a human
# choice. But the frame number plus or minus 15 frames is less subject to human bias.

# Use attention? bidirectional lstm?
# how many lstm layers to use? dont wanna overfit

def clamp(l, h, x):
  if x < l:
    return l
  if x > h:
    return h
  return x


NUMLABELS = 9


class mylstm(nn.Module):
  def __init__(self, hidden_dim, input_dim, output_size):
    super(mylstm, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim

    # context_seq = (len(frames), 1, hiddensize)
    # for each frame in frames:
    #   frame is [features for each vehicle] with shape (feature_seq_len, 1, insize)
    #   run encoder on the frame sequence to give shape (1, 1, hiddensize)
    #   set context_seq_i = encder output
    # compute lstm(context_seq)

    self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2)
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2)
    self.hidden2out = nn.Linear(hidden_dim, output_size)

  def forward(self, data):
    # see nn.LSTM documentation for input and output shapes
    context_seq = self.encoder(data) # == h_T, (hs, cs)
    context_seq = context_seq[1] # == (hs,cs)
    context_seq = context_seq[0] # == hs
    context_seq = context_seq[-1] # == hs for last layer, i think
    context_seq = context_seq.view(context_seq.shape[0], 1, context_seq.shape[1])
    lstm_out, _ = self.lstm(context_seq)
    out_space = self.hidden2out(lstm_out.view(lstm_out.shape[0], -1))
    out_scores = F.log_softmax(out_space, dim=1)
    return out_scores


def train(sequences):
  print('Training')
  model = mylstm(8*17, 17, NUMLABELS)
  model.to(torch.device('cuda'))
  if load_model:
    model.load_state_dict(torch.load('model1.pt'))
  model.train()

  # 0 : rightTO
  # 1 : lcRel
  # 2 : cutin
  # 3 : cutout
  # 4 : evtEnd
  # 5 : objTurnOff
  # 6 : end
  # 7 : barrier
  # 8 : NOLABEL
  class_weights = torch.tensor([1/1400,1/600,1/200,1/380,1/1200,1/50,1/170,1/90,1/8000],device=torch.device('cuda'))/0.04796432871394172
  # class_weights = torch.tensor([1400/4140,600/4140,200/4140,380/4140,1200/4140,50/4140,170/4140,90/4140,50/4140],device=torch.device('cuda'))

  # loss_function = nn.CrossEntropyLoss(weight=class_weight)
  loss_function = nn.NLLLoss(weight=class_weights)
  # optimizer = optim.SGD(model.parameters(), lr=0.1)
  optimizer = optim.Adam(model.parameters())

  print('Enter training loop')
  ema_loss = 1
  losses = []
  for epoch in range(5000):
    if epoch % 100 == 0 and epoch > 0:
      with torch.no_grad():
        model.eval()
        print('Saving model at epoch:',epoch)
        torch.save(model.state_dict(), 'model1.pt')
        for x, y in sequences[:20]:
          model.zero_grad()
          yhat = model(x)
          yhat = yhat.argmax(dim=1)
          print('yhat:', yhat)
          print('y   :', y)
        model.train()
        with open('losses.pkl', 'wb') as file:
          pickle.dump(losses, file)

    for i,(x, y) in enumerate(sequences):
      loss = loss_function(model(x), y)
      losses.append(float(loss))
      loss.backward()
      if i % 10 == 0: # use minibatches of size 10
          optimizer.step()
          model.zero_grad()

    ema_loss = .6 * ema_loss + .4 * losses[-1]
    print(epoch, '/', 5000, '  ', ema_loss)

  with open('losses.pkl', 'wb') as file:
    pickle.dump(losses, file)

  print('Finished training')
  model.eval()
  with torch.no_grad():
    for x, y in sequences:
      model.zero_grad()
      yhat = model(x)
      yhat = yhat.argmax(dim=1)
      print('yhat:', yhat)
      print('y   :', y)
      with open('predictions.pkl', 'wb') as file:
        pickle.dump(yhat.cpu().numpy().tolist(), file)

def test(sequences):
  model = mylstm(8 * 17, 17, NUMLABELS)
  model.to(torch.device('cuda'))
  model.load_state_dict(torch.load('model1.pt'))
  model.eval()
  with torch.no_grad():
    for x, y in sequences:
      model.zero_grad()
      yhat = model(x)
      yhat = yhat.argmax(dim=1)
      print('yhat:', yhat)
      print('y   :', y)
      with open('predictions.pkl', 'wb') as file:
        pickle.dump(yhat.cpu().numpy().tolist(), file)

if __name__ == '__main__':
  print('Loading data.')
  if make_new:
    # Get paths to precomputed features
    if not testing:
      filepaths = []
      for (dirpath, dirnames, filenames) in os.walk('precomputed/features'):
        filepaths.extend(dirpath + '/' + f for f in filenames)

      # Collect labels and features into one place
      data = []  # == [path, [label, framnum], features] where features == [rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs]
      for filepath in filepaths:
        labelsFilePath = filepath.replace('features/', 'convertedGroundTruthLabels/').replace('_m0.pkl', '_labels.txt')

        if not os.path.isfile(labelsFilePath):
          continue

        videodata = [filepath]

        # Get labels
        with open(labelsFilePath) as labelsFile:
          labels = []
          lines = labelsFile.readlines()
          labelLines = [line.rstrip('\n') for line in lines]
          for line in labelLines:
            label, labelTime = line.split(',')
            label = label.split('=')[0]
            frameNumber = int((float(labelTime) % 300) * 30)
            labels.append((label, frameNumber))
          videodata.append(labels)

        # Get features
        with open(filepath, 'rb') as featuresfile:
          videodata.append(list(pickle.load(featuresfile)))

        data.append(videodata)

      print('Loaded data')

      # Get all labels
      #AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'barrier', 'NOLABEL']
      AllPossibleLabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
      labels2Tensor = {}
      for label in AllPossibleLabels:
        labels2Tensor[label] = torch.tensor([len(labels2Tensor)])

      # Make input tensors from the data
      sequences = []
      for path, labels, features in data:

        # The data would be very skewed (almost always NOLABEL) if we just had one big sequence for the entire video
        # Instead train on smaller chunks of the video that contain some interesting action
        intervalwidth = 30 * 60
        for i in range(0, 30*60*5 - intervalwidth, intervalwidth):
          left = i
          right = i + intervalwidth

          frameNum2LabelTensors = {}
          for label, frameNum in labels:
            if frameNum in frameNum2LabelTensors:
              frameNum += 1
            frameNum2LabelTensors[frameNum] = labels2Tensor[label]

          # Now for each frame in the interval
          for label, frameNum in labels:
            if left <= frameNum <= right:
            #if True:
              xs, ys = [], []

              (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs) = features

              # Only look at data from this interval of frames
              vehicles = vehicles[i:i+intervalwidth+1]
              probs = boxcornerprobs[i:i+intervalwidth+1]
              lanescores = lanescores[i:i+intervalwidth+1]

              # for each frame in the interval
              for j, (vehicles, probs, scores) in enumerate(zip(vehicles, probs, lanescores)):
                scores = [float(score.cpu().numpy()) for score in scores]

                # for some reason I organized the pkl file s.t. we have to do this
                probsleft = probs[:len(vehicles)]  # left lane line probability map values at box corners for each vehicle
                probsright = probs[len(vehicles):]  # See LaneLineDetectorERFNet.py::175

                features_xs = []
                # Create tensors
                for vehicle, probleft, probright in zip(vehicles, probsleft, probsright):
                  # Put object id into sensible range
                  objectid, x1, y1, x2, y2 = vehicle
                  objectid = (objectid % 1000) / 1000
                  vehicle = (objectid, x1, y1, x2, y2)

                  features_xs.append(torch.tensor([[*vehicle, *probleft, *probright, *scores]], dtype=torch.float))
                if len(features_xs) == 0:
                  features_xs = [torch.tensor([ [0]*13 + [*scores] ], dtype=torch.float)] # make sure there is always an input tensor
                features_xs = torch.cat(features_xs)

                xs.append(features_xs)
                if i+j in frameNum2LabelTensors:
                  ys.append(frameNum2LabelTensors[i+j])
                else:
                  ys.append(labels2Tensor['8'])

              xs = pad_sequence(xs).to(torch.device('cuda'))
              xs = torch.flip(xs,dims=(0,))
              ys = torch.cat(ys).to(torch.device('cuda'))
              sequences.append((xs, ys))

              # There may be more than one label in this interval.
              # If we ran this loop twice in this interval then we would append the same exact (xs,ys) to sequences
              break
    else:
      featureFilePath = r"D:\features\Gen5_RU_2019-10-07_07-56-42-0001_m0.pkl"
      labelsFilePath = "precomputed/convertedGroundTruthLabels/Gen5_RU_2019-10-07_07-56-42-0001_labels.txt"
      filepaths = []
      filepaths.append(featureFilePath)

      # Collect labels and features into one place
      data = []  # == [path, [label, framnum], features] where features == [rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs]
      if not os.path.isfile(labelsFilePath):
        print("invalid test file")

      videodata = [featureFilePath]

      # Get labels
      with open(labelsFilePath) as labelsFile:
        labels = []
        lines = labelsFile.readlines()
        labelLines = [line.rstrip('\n') for line in lines]
        for line in labelLines:
          label, labelTime = line.split(',')
          label = label.split('=')[0]
          frameNumber = int((float(labelTime) % 300) * 30)
          labels.append((label, frameNumber))
        videodata.append(labels)

      # Get features
      with open(featureFilePath, 'rb') as featuresfile:
        videodata.append(list(pickle.load(featuresfile)))

      data.append(videodata)

      print('Loaded data')

      # Get all labels
      # AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'barrier', 'NOLABEL']
      AllPossibleLabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
      labels2Tensor = {}
      for label in AllPossibleLabels:
        labels2Tensor[label] = torch.tensor([len(labels2Tensor)])

      # Make input tensors from the data
      sequences = []
      for path, labels, features in data:

        # The data would be very skewed (almost always NOLABEL) if we just had one big sequence for the entire video
        # Instead train on smaller chunks of the video that contain some interesting action
        intervalwidth = 30 * 60
        for i in range(0, 30 * 60 * 5 - intervalwidth, intervalwidth):
          left = i
          right = i + intervalwidth

          frameNum2LabelTensors = {}
          for label, frameNum in labels:
            if frameNum in frameNum2LabelTensors:
              frameNum += 1
            frameNum2LabelTensors[frameNum] = labels2Tensor[label]

          # Now for each frame in the interval
          for label, frameNum in labels:
            if left <= frameNum <= right:
              # if True:
              xs, ys = [], []

              (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs) = features

              # Only look at data from this interval of frames
              vehicles = vehicles[i:i + intervalwidth + 1]
              probs = boxcornerprobs[i:i + intervalwidth + 1]
              lanescores = lanescores[i:i + intervalwidth + 1]

              # for each frame in the interval
              for j, (vehicles, probs, scores) in enumerate(zip(vehicles, probs, lanescores)):
                scores = [float(score.cpu().numpy()) for score in scores]

                # for some reason I organized the pkl file s.t. we have to do this
                probsleft = probs[
                            :len(vehicles)]  # left lane line probability map values at box corners for each vehicle
                probsright = probs[len(vehicles):]  # See LaneLineDetectorERFNet.py::175

                features_xs = []
                # Create tensors
                for vehicle, probleft, probright in zip(vehicles, probsleft, probsright):
                  # Put object id into sensible range
                  objectid, x1, y1, x2, y2 = vehicle
                  objectid = (objectid % 1000) / 1000
                  vehicle = (objectid, x1, y1, x2, y2)

                  features_xs.append(torch.tensor([[*vehicle, *probleft, *probright, *scores]], dtype=torch.float))
                if len(features_xs) == 0:
                  features_xs = [torch.tensor([[0] * 13 + [*scores]],
                                              dtype=torch.float)]  # make sure there is always an input tensor
                features_xs = torch.cat(features_xs)

                xs.append(features_xs)
                if i + j in frameNum2LabelTensors:
                  ys.append(frameNum2LabelTensors[i + j])
                else:
                  ys.append(labels2Tensor['8'])

              xs = pad_sequence(xs).to(torch.device('cuda'))
              xs = torch.flip(xs, dims=(0,))
              ys = torch.cat(ys).to(torch.device('cuda'))
              sequences.append((xs, ys))

              # There may be more than one label in this interval.
              # If we ran this loop twice in this interval then we would append the same exact (xs,ys) to sequences
              break

    print(len(sequences))
    with open('tensors.pkl', 'wb') as file:
      pickle.dump(sequences, file)

  else:
    with open('tensors.pkl', 'rb') as file:
      sequences = pickle.load(file)
    if not testing:
      train(sequences)
    else:
      test(sequences)