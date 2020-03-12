import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

# 0 : rightTO
# 1 : lcRel
# 2 : cutin
# 3 : cutout
# 4 : evtEnd
# 5 : objTurnOff
# 6 : end
# 7 : barrier
# 8 : NOLABEL

# TODO convert the object ID into a categorical input? onehot? modulo?
# TODO sort the input tensors by objectid?
# TODO use minibatch?
# TODO regularization?

# Doing input summarization inline with label predictions inflates the
# number of timesteps the network has to work across.
# If there were less timesteps then we could increase the time window
# that the network has access to

# It would be nice if we could increase the window of the label predictions
# A problem is that having to choose the correct label on the precise frame number
# is too demanding of the network. The exact frame number is noisy and is a human
# choice. But the frame number plus or minus 15 frames is less subject to human bias.

# Use attention?
# two lstms. an encoder that takes bboxes and other features aand outputs a single fixed size vector.
# and the other that has 1 timestep per frame of video and consumes the encoder's output


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

    # The LSTM takes word embeddings as inputs, and outputs hidden states
    # with dimensionality hidden_dim.
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=4,)

    # The linear layer that maps from hidden state space to tag space
    self.hidden2out = nn.Linear(hidden_dim, output_size)

  def forward(self, inputs):
    # add the batch dimension
    lstm_out, _ = self.lstm(inputs)
    # lstm_out, self.hidden = self.lstm(inputs, self.hidden)
    out_space = self.hidden2out(lstm_out.view(len(inputs), -1))
    out_scores = F.log_softmax(out_space, dim=1)
    return out_scores


def train(sequences):
  print('Training')
  model = mylstm(2*17, 17, NUMLABELS)
  model.to(torch.device('cuda'))
  loss_function = nn.NLLLoss()
  # optimizer = optim.SGD(model.parameters(), lr=0.1)
  optimizer = optim.Adam(model.parameters())

  tensors = sequences[0][0].to(torch.device('cuda'))
  labels = sequences[0][1].to(torch.device('cuda'))

  print('Get initial model outputs')
  # See what the scores are before training
  # Note that element i,j of the output is the score for tag j for word i.
  # Here we don't need to train, so the code is wrapped in torch.no_grad()
  with torch.no_grad():
    model.eval()
    yhat = model(tensors)
    model.train()

  print('Enter training loop')
  loss = None
  losses = []
  for epoch in range(5000):
    print(epoch, '/', 5000, '  ', loss)

    if epoch % 1000 == 999:
      with torch.no_grad():
        model.eval()
        print('Saving model at epoch:',epoch)
        torch.save(model.state_dict(), 'model.pt')
        for x, y in sequences:
          x = x.to(torch.device('cuda'))
          y = y.to(torch.device('cuda'))
          # Step 1. Remember that Pytorch accumulates gradients.
          # We need to clear them out before each instance
          model.zero_grad()

          # Step 3. Run our forward pass.
          yhat = model(x)
          yhat = yhat.argmax(dim=1)
          print('yhat:', yhat)
          print('y   :', y)

    model.train()
    for x, y in sequences:
      x = x.to(torch.device('cuda'))
      y = y.to(torch.device('cuda'))
      # Step 1. Remember that Pytorch accumulates gradients.
      # We need to clear them out before each instance
      model.zero_grad()

      # Step 3. Run our forward pass.
      yhat = model(x)

      # Step 4. Compute the loss, gradients, and update the parameters by
      #  calling optimizer.step()
      loss = loss_function(yhat, y)
      losses.append(float(loss))
      loss.backward()
      optimizer.step()

  with open('losses.pkl', 'wb') as file:
    pickle.dump(losses, file)

  print('Finished training')
  model.eval()
  with torch.no_grad():
    for x, y in sequences:
      x = x.to(torch.device('cuda'))
      y = y.to(torch.device('cuda'))
      # Step 1. Remember that Pytorch accumulates gradients.
      # We need to clear them out before each instance
      model.zero_grad()

      # Step 3. Run our forward pass.
      yhat = model(x)
      yhat = yhat.argmax(dim=1)
      print('yhat:', yhat)
      print('y   :', y)
      with open('predictions.pkl', 'wb') as file:
        pickle.dump(yhat.cpu().numpy().tolist(), file)


if __name__ == '__main__':
  print('Loading data.')

  # # Get paths to precomputed features
  # filepaths = []
  # for (dirpath, dirnames, filenames) in os.walk('precomputed/features'):
  #   filepaths.extend(dirpath + '/' + f for f in filenames)
  #
  # # Collect labels and features into one place
  # data = []  # == [path, [label, framnum], features] where features == [rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs]
  # for filepath in filepaths:
  #   videodata = [filepath]
  #
  #   # Get labels
  #   labelsFilePath = filepath.replace('features/', 'groundTruthLabels/').replace('_m0.pkl', '_labels.txt')
  #   with open(labelsFilePath) as labelsFile:
  #     labels = []
  #     lines = labelsFile.readlines()
  #     labelLines = [line.rstrip('\n') for line in lines]
  #     for line in labelLines:
  #       label, labelTime = line.split(',')
  #       label = label.split('=')[0]
  #       frameNumber = int((float(labelTime) % 300) * 30)
  #       labels.append((label, frameNumber))
  #     videodata.append(labels)
  #
  #   # Get features
  #   with open(filepath, 'rb') as featuresfile:
  #     videodata.append(list(pickle.load(featuresfile)))
  #
  #   data.append(videodata)
  #
  # print('Loaded data')
  #
  # # Get all labels
  # AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'barrier', 'NOLABEL']
  # labels2Tensor = {}
  # for label in AllPossibleLabels:
  #   labels2Tensor[label] = torch.tensor([len(labels2Tensor)])
  #
  # ENDSIGNAL = torch.tensor([[[-1]*17]], dtype=torch.float)  # size = (1,1,17)
  #
  # # Make input tensors from the data
  # sequences = []
  # for path, labels, features in data:
  #
  #   # The data would be very skewed (almost always NOLABEL) if we just had one big sequence for the entire video
  #   # Instead train on smaller chunks of the video that contain some interesting action
  #   intervalwidth = 30 * 2
  #   for i in range(0, 30*60*5 - intervalwidth, intervalwidth):
  #     left = i
  #     right = i + intervalwidth
  #
  #     # TODO i make the assumption that there is only 1 label per frame
  #     frameNum2LabelTensors = {}
  #     for label, frameNum in labels:
  #       frameNum2LabelTensors[frameNum] = labels2Tensor[label]
  #
  #     # Now for each frame in the interval
  #     for label, frameNum in labels:
  #       if left <= frameNum <= right:
  #         xs, ys = [], []
  #
  #         (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs) = features
  #
  #         # Only look at data from this interval of frames
  #         vehicles = vehicles[i:i+intervalwidth+1]
  #         probs = boxcornerprobs[i:i+intervalwidth+1]
  #         lanescores = lanescores[i:i+intervalwidth+1]
  #
  #         for j, (vehicles, probs, scores) in enumerate(zip(vehicles, probs, lanescores)):
  #           scores = [float(score.cpu().numpy()) for score in scores]
  #
  #           # for some reason I organized the pkl file s.t. we have to do this
  #           probsleft = probs[:len(vehicles)]  # left lane line probability map values at box corners for each vehicle
  #           probsright = probs[len(vehicles):]  # See LaneLineDetectorERFNet.py::175
  #
  #           # Create tensors
  #           for vehicle, probleft, probright in zip(vehicles, probsleft, probsright):
  #             # Put object id into sensible range
  #             objectid, x1, y1, x2, y2 = vehicle
  #             objectid = (objectid % 1000) / 1000
  #             vehicle = (objectid, x1, y1, x2, y2)
  #
  #             xs.append(torch.tensor([[[*vehicle, *probleft, *probright, *scores]]], dtype=torch.float))
  #             ys.append(labels2Tensor['NOLABEL'])
  #
  #           # entice the network to produce a label
  #           xs.append(ENDSIGNAL)
  #           if i+j in frameNum2LabelTensors:
  #             ys.append(frameNum2LabelTensors[i+j])
  #           else:
  #             ys.append(labels2Tensor['NOLABEL'])
  #
  #         xs = torch.cat(xs).to(torch.device('cuda'))
  #         ys = torch.cat(ys).to(torch.device('cuda'))
  #         sequences.append((xs, ys))
  #
  #         # There may be more than one label in this interval.
  #         # If we ran this loop twice in this interval then we would append the same exact (xs,ys) to sequences
  #         break
  #
  # print(len(sequences))
  # with open('tensors.pkl', 'wb') as file:
  #   pickle.dump(sequences, file)
  with open('tensors.pkl', 'rb') as file:
    sequences = pickle.load(file)
  train(sequences)
