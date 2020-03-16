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

PRECOMPUTE = False

# Get all labels
AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'NOLABEL']
labels2Tensor = {}
for label in AllPossibleLabels:
  labels2Tensor[label] = torch.tensor([len(labels2Tensor)])

NUMLABELS = len(AllPossibleLabels)
PREDICT_EVERY_NTH_FRAME = 8
WINDOWWIDTH = 30*16
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

class mylstm(nn.Module):
  def __init__(self, hidden_dim, input_dim, output_dim):
    super(mylstm, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.encoder = nn.LSTM(input_dim, hidden_dim)
    self.pencoder1 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True)
    self.pencoder2 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True)
    self.pencoder3 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True)
    self.pencoder4 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True)
    self.attn = nn.Linear(hidden_dim + output_dim, WINDOWWIDTH//16)
    self.attn_combine = nn.Linear(hidden_dim*2 + output_dim, hidden_dim)
    self.dropout = nn.Dropout(.1)
    self.gru = nn.GRU(hidden_dim, hidden_dim)
    self.out = nn.Linear(hidden_dim, output_dim)
    # self.out = nn.Linear(hidden_dim*2, output_dim)

  def forward(self, data):
    # see nn.LSTM documentation for input and output shapes
    # data = (seqlen=numBoundingBoxesInFrame, 240, 17)
    # print(data.shape)
    _, context_seq = self.encoder(data)
    # print(context_seq[0].shape)
    context_seq = context_seq[0].view(data.shape[1]//2,1,-1)
    # print(context_seq.shape)
    context_seq, _ = self.pencoder1(context_seq)
    context_seq = context_seq.view(data.shape[1]//4,1,-1)
    context_seq, _ = self.pencoder2(context_seq)
    context_seq = context_seq.view(data.shape[1]//8,1,-1)
    context_seq, _  = self.pencoder3(context_seq)
    context_seq = context_seq.view(data.shape[1]//16,1,-1)
    context_seq, hidden  = self.pencoder4(context_seq)
    context_seq = context_seq.view(1,WINDOWWIDTH//16,self.hidden_dim*2)
    # return F.log_softmax(self.out(context_seq), dim=1)[0]

    hidden = hidden[0][0].view(1,1,-1)
    input = torch.ones((1,self.output_dim),device=torch.device('cuda')) / self.output_dim
    outputs = torch.zeros((WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME, NUMLABELS),device=torch.device('cuda'))

    # print(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)
    for i in range(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME):
      input = self.dropout(input)
      attn_weights = F.softmax(self.attn(torch.cat((input, hidden[0]), 1)), dim=1)
      # print(attn_weights.shape)
      attn_applied = torch.bmm(attn_weights.unsqueeze(0), context_seq)

      # print(attn_applied.shape)
      # print(input.shape)
      # print(hidden.shape)
      output = torch.cat((input, attn_applied[0]), 1)
      # print(output.shape)
      output = self.attn_combine(output).unsqueeze(0)

      # print(output.shape)
      # print()
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
      outputs[i] += F.log_softmax(self.out(output[0]), dim=1)[0]
      input = outputs[i].view(1,NUMLABELS)
    return outputs


def evaluate(model, loss_function, sequences, saveFileName, n=None):
  outputs = []
  average_loss = 0
  for x, y in sequences[:n]:
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


def checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences,n=None):
  model.eval()
  with torch.no_grad():
    print('Saving model at epoch:', epoch)
    if n is None:
      torch.save(model.state_dict(), 'model.pt')
      with open('losses.pkl', 'wb') as file:
        pickle.dump(losses, file)
    evaluate(model, loss_function, trainSequences, 'trainOutputs.txt',n)
    evaluate(model, loss_function, testSequences, 'testOutputs.txt',n)
    print('Saved', epoch)
  model.train()


def train(trainSequences, testSequences):
  print('Training')
  model = mylstm(60, 17, NUMLABELS)
  model.to(torch.device('cuda'))
  model.train()

  class_weights = [1, # rightTO
                   1, # lcRel
                   1, # cutin
                   1, # cutout
                   1, # evtEnd
                   1, # objTurnOff
                   1, # end
                   1/22]# NOLABEL
  class_weights = torch.tensor(class_weights, device=torch.device('cuda'))/sum(class_weights)

  loss_function = nn.CrossEntropyLoss(weight=class_weights)
  # loss_function = nn.NLLLoss(weight=class_weights)
  # optimizer = optim.SGD(model.parameters(), lr=0.1)
  optimizer = optim.Adam(model.parameters(), lr=.01)

  print('Enter training loop')
  ema_loss = None
  losses = []
  for epoch in range(5000):
    mloss = 0
    for i in range(len(trainSequences)):
      k = random.randint(0,len(trainSequences)-1)
      x,y = trainSequences[k]
      loss = loss_function(model(x), y) / 3
      loss.backward()
      mloss += loss
      if i % 3 == 2:
        optimizer.step()
        model.zero_grad()
        losses.append(float(mloss))
        mloss = 0

    if epoch % 2 == 0 and epoch > 0:
      if epoch % 10 == 0:
        checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences)
      else:
        checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences,10)
    if ema_loss is None:
      ema_loss = losses[-1]
    ema_loss = .6 * ema_loss + .4 * losses[-1]
    print(epoch, '/', 5000, '  ', ema_loss)

  print('Finished training')
  checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences)


def getSequenceForInterval(low, high, features, labels, frameNum2LabelTensors):
  xs, ys = [], []

  # (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs) = features
  (_, lanescores, vehicles, boxcornerprobs) = features

  # Only look at data from this interval of frames
  vehicles = vehicles[low:high]
  probs = boxcornerprobs[low:high]
  lanescores = lanescores[low:high]
  # lines = lines[low:high]
  # print(len(lines))
  # print(len(lanescores))
  # print(len(probs))
  # print(len(vehicles))
  # print()
  # print()
  # print()

  # for the jth frame in the interval
  for j, (vehicles_, probs_, lanescores_) in enumerate(zip(vehicles, probs, lanescores)):
    lanescores_ = [float(score.cpu().numpy()) for score in lanescores_]

    # # Try to get some info from the lane lines
    # lline = lines_[0]
    # rline = lines_[1]
    # print(lline)
    # try:
    #     for i in range(len(lline)):
    #         lline[i] = lline[i][0]
    #     for i in range(len(rline)):
    #         rline[i] = rline[i][0]
    #     lline = lline[::30][:5]
    #     rline = rline[::30][:5]
    #     if len(lline) < 5: lline += [0] * (5-len(lline))
    #     if len(rline) < 5: rline += [0] * (5-len(rline))
    # except:
    #     continue

    # for some reason I organized the pkl file s.t. we have to do this
    probsleft = probs_[:len(vehicles_)]  # left lane line probability map values at box corners for each vehicle
    probsright = probs_[len(vehicles_):]  # See LaneLineDetectorERFNet.py::175

    features_xs = []

    # print(len(probs_))
    # print(len(lines_[0]))
    # print(len(lines_[1]))
    # print(len(vehicles_))
    # print(len(probsleft))
    # print(len(probsright))

    # sort by objectid
    stuff = sorted(list(zip(vehicles_, probsleft, probsright)), key=lambda x:-x[0][0])
    for vehicle, probleft, probright in stuff:
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


def getSequencesForFiles(files):
  # returns a list of (path, [label, framnum], features)
  # where features == (rawboxes, boxscores, lines, lanescores, vehicles, boxcornerprobs)
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
    frameNum2LabelTensors = {}
    for label, frameNum in labels:
      while frameNum//PREDICT_EVERY_NTH_FRAME in frameNum2LabelTensors:
        frameNum += PREDICT_EVERY_NTH_FRAME
      frameNum2LabelTensors[frameNum//PREDICT_EVERY_NTH_FRAME] = labels2Tensor[label]

    # Then do features tensors
    for i in range(0, 30*60*5 - WINDOWWIDTH, WINDOWSTEP):
      [low, high] = [i, i + WINDOWWIDTH]
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

    with open('tensors.pkl', 'wb') as file:
      pickle.dump((trainSequences, testSequences), file)
    print('Wrote tensors pickle. Exiting.')
  else:
    with open('tensors.pkl', 'rb') as file:
      (trainSequences, testSequences) = pickle.load(file)
    train(trainSequences, testSequences)
