# Note:
#   smaller batch size seems to make the tesitng accuracy better.

import os, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from Dataset import Dataset

torch.manual_seed(1209809284)
device = torch.device('cuda')
devcount = torch.cuda.device_count()
print('Device count:',devcount)
torch.cuda.set_device(devcount-1)
print('Current device:',torch.cuda.current_device())
print('Device name:',torch.cuda.get_device_name(devcount-1))

AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'NOLABEL']
NUMLABELS = len(AllPossibleLabels)

PREDICT_EVERY_NTH_FRAME = 12
WINDOWWIDTH = 30*12
INPUT_FEATURE_DIM = 43

BATCH_SIZE = 16
HIDDEN_DIM = 256
DROPOUT_RATE = .8
TEACHERFORCING_RATIO = 8 # to 1

RESUME_TRAINING = False
LOAD_CHECKPOINT_PATH = 'mostrecent.pt'

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

def oneHot(i):
  return torch.tensor([[[0. if j != i else 1. for j in range(NUMLABELS)]]], device=device)
oneHot = [oneHot(i) for i in range(len(AllPossibleLabels))]

class Losses:
  def __init__(self):
    self.trainLoss = []
    self.testLoss = []
    self.trainAcc = []
    self.testAcc = []
    self.trainAcc2 = []
    self.testAcc2 = []
    self.trainIou = []
    self.testIou = []

def augment(t):
  tens = t.clone()
  i = random.randint(0,2)
  if i == 0:
    tens[0,2], tens[0,4] = tens[0,4], tens[0,2]
    tens[0,2] *= -1
    tens[0,4] *= -1
    tens[0,8], tens[0,9] = tens[0,9], tens[0,8]
    tens[0,8] *= -1
    tens[0,9] *= -1
    tens[0,13], tens[0,14] = tens[0,14], tens[0,13]
    tens[0,16], tens[0,18] = tens[0,18], tens[0,16]
    tens[0,-20:-10], tens[0,-10:] = tens[0,-10:], tens[0,-20:-10]
    tens[0,-20:] *= -1
  elif i == 1:
    tens[0,2:13] += torch.randn_like(tens[0,2:13]) * .2
    tens[0,-20:] += torch.randn_like(tens[0,-20:]) * .2
  else:
    tens += torch.randn_like(tens) * .1
  return tens

def getSequentialBatch(dataset, aug=True, size=BATCH_SIZE):
  batch_xs,batch_xlengths,batch_ys = [],[],[]
  N = len(dataset)
  k = random.randint(0,N-1)
  for i in range(4*size):
    ((xs,xlengths),ys) = dataset[(i//4+k)%N]
    if i % 4:
      xs = [augment(x) for x in xs]
    batch_xs.extend(xs)
    batch_xlengths.extend(xlengths)
    batch_ys.extend(ys)
  batch_xs = pad_sequence(batch_xs).to(device=device)
  batch_xlengths = torch.tensor(batch_xlengths, device=device)
  batch_ys = torch.tensor(batch_ys, device=device)
  return (batch_xs,batch_xlengths),batch_ys

def getBatch(dataset, aug=True, size=BATCH_SIZE):
  batch_xs,batch_xlengths,batch_ys = [],[],[]
  N = len(dataset)
  for i in range(size):
    ((xs,xlengths),ys) = dataset[random.randint(0,N-1)]
    if aug and random.randint(0,5) == 2:
      xs = [augment(x) for x in xs]
    batch_xs.extend(xs)
    batch_xlengths.extend(xlengths)
    batch_ys.extend(ys)
  batch_xs = pad_sequence(batch_xs).to(device=device)
  batch_xlengths = torch.tensor(batch_xlengths, device=device)
  batch_ys = torch.tensor(batch_ys, device=device)
  return (batch_xs,batch_xlengths),batch_ys

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.encoder = nn.LSTM(INPUT_FEATURE_DIM, HIDDEN_DIM)

    self.pencoder1 = nn.LSTM(HIDDEN_DIM*2, HIDDEN_DIM, bidirectional=True, batch_first=True)
    self.dropout1 = nn.Dropout(p=DROPOUT_RATE)

    self.pencoder2 = nn.LSTM(HIDDEN_DIM*2, HIDDEN_DIM, bidirectional=True, batch_first=True)
    self.dropout2 = nn.Dropout(p=DROPOUT_RATE)

    self.pencoder3 = nn.LSTM(HIDDEN_DIM*2, HIDDEN_DIM, bidirectional=True, batch_first=True)
    self.dropout3 = nn.Dropout(p=DROPOUT_RATE)

    self.norm = nn.LayerNorm(HIDDEN_DIM)
    self.mlp1 = nn.Linear(HIDDEN_DIM,HIDDEN_DIM)
    self.mlp2 = nn.Linear(HIDDEN_DIM,HIDDEN_DIM)

    self.rnn = nn.LSTM(NUMLABELS + HIDDEN_DIM, HIDDEN_DIM, num_layers=2, batch_first=True)
    self.out = nn.Linear(2*HIDDEN_DIM, NUMLABELS)

  def encode(self, data):
    xs, xlengths = data

    batch_size = xs.shape[1] // WINDOWWIDTH
    packed_padded = pack_padded_sequence(xs, xlengths, enforce_sorted=False)
    packed_padded_out, hidden = self.encoder(packed_padded)
    # Check unpacked_lengths against xlengths to verify correct output ordering
    # unpacked_padded, unpacked_lengths = pad_packed_sequence(packed_padded_hidden[0])

    seq = hidden[0].view(batch_size, WINDOWWIDTH, HIDDEN_DIM)

    seq = seq.reshape(batch_size, WINDOWWIDTH // 2, 2 * HIDDEN_DIM)
    seq = self.dropout1(seq)
    seq, _ = self.pencoder1(seq)
    # Average forward and backward dims
    seq = seq.view(batch_size, WINDOWWIDTH // 2, 2, HIDDEN_DIM).sum(dim=2) / 2

    seq = seq.reshape(batch_size, WINDOWWIDTH // 4, 2 * HIDDEN_DIM)
    seq = self.dropout2(seq)
    seq, _ = self.pencoder2(seq)
    seq = seq.view(batch_size, WINDOWWIDTH // 4, 2, HIDDEN_DIM).sum(dim=2) / 2

    seq = seq.reshape(batch_size, WINDOWWIDTH // 8, 2 * HIDDEN_DIM)
    seq = self.dropout3(seq)
    seq, _ = self.pencoder3(seq)
    seq = seq.view(batch_size, WINDOWWIDTH // 8, 2, HIDDEN_DIM).sum(dim=2) / 2

    seq = self.norm(seq)

    seq = seq.view(batch_size * WINDOWWIDTH // 8, HIDDEN_DIM)
    seq = F.tanh(self.mlp1(seq))
    seq = seq.view(batch_size, WINDOWWIDTH // 8, HIDDEN_DIM)

    return seq

  # https://stackoverflow.com/questions/54242123/order-of-layers-in-hidden-states-in-pytorch-gru-return
  def decoderStep(self, yi_1, ci_1, hidden, H):
    si, hidden = self.rnn(torch.cat([yi_1, ci_1], dim=2), hidden)

    si = F.tanh(self.mlp2(si))

    scores = torch.bmm(si, H.transpose(1,2)) # Bx1xT
    weights = F.softmax(scores, dim=2) # Bx1xT
    ci = torch.bmm(weights, H) # Bx1x2H
    yi = F.softmax(self.out(torch.cat([si, ci], dim=2)), dim=2) # Bx1xO
    return yi, ci, hidden

  def forward(self, xs:torch.Tensor, gtys:torch.Tensor=None):
    batch_size = xs[0].shape[1] // WINDOWWIDTH
    if gtys is not None:
      gtys = gtys.view(batch_size, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)

    seq = self.encode(xs)
    yi = torch.zeros((batch_size,1,NUMLABELS), device=device)
    ci = torch.zeros((batch_size,1,HIDDEN_DIM), device=device)
    hidden = None

    ys = []
    for i in range(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME):
      yi, ci, hidden = self.decoderStep(yi, ci, hidden, seq)
      ys.append(yi)
      if gtys is not None:
        yi = torch.cat([oneHot[gtys[j,i].item()] for j in range(batch_size)])

    ys = torch.cat(ys, dim=1)
    ys = ys.view(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME * batch_size, NUMLABELS)
    ys = torch.log(ys)
    return ys

  def beamDecode(self, data:torch.Tensor):
    seq = self.encode(data)
    beams = [] # tuple of (outputs, previous hidden, next yi, beam log prob)
    batch_size = 1
  
    # get the initial beam
    yi = torch.zeros((batch_size,1,NUMLABELS), device=device)
    ci = torch.zeros((batch_size,1,HIDDEN_DIM), device=device)
    hidden = None

    yi, ci, hidden = self.decoderStep(yi, ci, hidden, seq)
    for i in range(NUMLABELS):
      beams.append(([i], oneHot[i], ci, hidden, math.log(float(yi[0,0,i]))))
    for i in range(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME - 1):
      newBeams = []
      for beam in beams:
        ys, yi, ci, hidden, beamLogProb = beam
        yi, ci, hidden = self.decoderStep(yi, ci, hidden, seq)
        for i in range(NUMLABELS):
          newBeam = (ys + [i], oneHot[i], ci, hidden, beamLogProb + math.log(float(yi[0,0,i])))
          newBeams.append(newBeam)
      beams = sorted(newBeams, key=lambda x:-x[-1])[:NUMLABELS]
    ys, _, _, _, _ = beams[0]
    return np.array([ys])

def evaluate(model, lossFunction, sequences, saveFileName):
  start = time.time()

  outputs = []
  avgacc1 = 0
  avgacc2 = 0
  avgiou = 0

  SAMPLES = 16

  for i in range(SAMPLES):
    xs, ys = getBatch(sequences)
    yhats = model(xs).view(BATCH_SIZE, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME, NUMLABELS)
    yhats = yhats.argmax(dim=2).cpu().numpy()
    ys = ys.view(BATCH_SIZE, WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME).cpu().numpy()
    for j in range(BATCH_SIZE):
      pred = ['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in yhats[j].tolist()]
      exp = ['.' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in ys[j].tolist()]
      a = set(yhats[j].tolist())
      b = set(ys[j].tolist())
      avgiou += len(a&b)/len(a|b) / BATCH_SIZE

      outputs.append(''.join(pred) + ' ' + ''.join(exp) + '\n\n')
    numlabels = (ys != AllPossibleLabels.index('NOLABEL')).sum()
    if numlabels > 0:
      avgacc1 += ((yhats == ys) & (ys != AllPossibleLabels.index('NOLABEL'))).sum() / numlabels
    avgacc2 += (yhats == ys).sum() / (BATCH_SIZE * WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME)

  ## Compare beam search with raw outputs
  #batch_size = 1
  #for i in range(8):
  #  xs, ys = getBatch(sequences, aug=False, size=batch_size)
  #  yhats = model(xs).view(batch_size, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME, NUMLABELS)
  #  yhats = yhats.argmax(dim=2).cpu().numpy()
  #  ys = ys.view(batch_size, WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME).cpu().numpy()
  #  pred = ['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in yhats[0].tolist()]
  #  exp = ['.' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in ys[0].tolist()]
  #  output = ''.join(pred) + ' ' + ''.join(exp) + ' '
  #  yhats = model.beamDecode(xs)
  #  pred = ['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in yhats[0].tolist()]
  #  output = output +  ''.join(pred) + ' beam\n\n'
  #  outputs.append(output)

  end = time.time()
  outputs = ['acc:'+str(avgacc1/SAMPLES)+' iou:'+str(avgiou/SAMPLES)+' evaltime:'+str(int(end-start))+'\n\n'] + outputs

  return avgacc1 / SAMPLES, avgacc2 / SAMPLES, avgiou / SAMPLES, outputs

def checkpoint(trainloss,losses, model, optimizer, lossFunction, trainData, testData):
  model.eval()
  with torch.no_grad():
    avgtrainacc, avgtrainacc2, avgtrainiou, trainOuts = evaluate(model, lossFunction, trainData, 'trainOutputs.txt')
    avgtestacc, avgtestacc2, avgtestiou, testOuts = evaluate(model, lossFunction, testData, 'testOutputs.txt')
    if len(losses.testAcc) and avgtestacc > max(losses.testAcc):
      torch.save((model.state_dict(), optimizer.state_dict()), 'maxacc.pt')
      torch.save(losses, 'maxacc_losses.pt')
      with open('outputsBestAcc.txt','w') as f:
        f.writelines(testOuts)
    if len(losses.testIou) and avgtestiou > max(losses.testIou):
      torch.save((model.state_dict(), optimizer.state_dict()), 'maxiou.pt')
      torch.save(losses, 'maxiou_losses.pt')
      with open('outputsBestIou.txt','w') as f:
        f.writelines(testOuts)
    with open('outputs.txt','w') as f:
      f.writelines(testOuts)
    torch.save((model.state_dict(), optimizer.state_dict()), 'mostrecent.pt')
    torch.save(losses, 'mostrecent_losses.pt')
    losses.trainLoss.append(trainloss)
    losses.trainAcc.append(avgtrainacc)
    losses.testAcc.append(avgtestacc)
    losses.trainAcc2.append(avgtrainacc2)
    losses.testAcc2.append(avgtestacc2)
    losses.trainIou.append(avgtrainiou)
    losses.testIou.append(avgtestiou)
  model.train()
  return avgtestacc, avgtestiou

def train(trainData, testData):
  (_,_,classCounts) = trainData.getStats()

  print('Training')
  model = Model()
  model.to(device)
  print(model)

  N = len(trainData)
  print('Train set num data:',N)
  print('Test set num data:',len(testData))
  print('Class counts:')
  for label,count in classCounts.items():
    print('\t',label,':',count)

  print('classCounts:',classCounts)
  classWeights = [1/(classCounts[lab]+1) for lab in range(NUMLABELS)] # order is important here
  classWeights[-1] *= 2
  classWeights = torch.tensor(classWeights, device=device) / sum(classWeights)

  lossFunction = nn.NLLLoss(weight=classWeights)
  optimizer = optim.Adam(model.parameters(), lr=.0002)

  losses = Losses()

  if RESUME_TRAINING:
    print('Resuming from checkpoint')

    (model_state, optimizer_state) = torch.load(LOAD_CHECKPOINT_PATH)

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad()

  model.train()

  print('Enter training loop')
  avgloss = 0
  now = time.time()
  prev = now
  i = 0
  while True:
    xs,ys = getBatch(trainData)
    if i % TEACHERFORCING_RATIO:
      loss = lossFunction(model(xs,ys), ys)
    else:
      loss = lossFunction(model(xs), ys)
    avgloss += .1 * (float(loss) - avgloss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    i += 1

    now = time.time()
    if now - prev > 60:
      testacc, testiou = checkpoint(avgloss, losses, model, optimizer, lossFunction, trainData, testData)
      print('trainloss {:1.5} i/s {:1.5} testacc {:1.5} testiou {:1.5}'.format(avgloss, i/60, testacc, testiou))
      i = 0
      prev = time.time()

if __name__ == '__main__':
  trainData = Dataset(loadPath='trainDataset.pkl')
  testData = Dataset(loadPath='testDataset.pkl')
  train(trainData, testData)

