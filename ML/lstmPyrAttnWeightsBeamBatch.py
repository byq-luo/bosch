import os, time, pickle, random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1209809284)
device = torch.device('cuda')

# TODO try to think of how to engineer more rich features.
# TODO consider using pytorch's DataSet and DataParallel if it speeds up training to use multigpus.

AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'objTurnOff', 'end', 'NOLABEL']
labels2Tensor = {}
for label in AllPossibleLabels:
  labels2Tensor[label] = torch.tensor([len(labels2Tensor)])

NUMLABELS = len(AllPossibleLabels)

PREDICT_EVERY_NTH_FRAME = 10
WINDOWWIDTH = 30*8
WINDOWSTEP = WINDOWWIDTH // 10

BATCH_SIZE = 64
INPUT_FEATURE_DIM = 36
HIDDEN_DIM = 512
DROPOUT_RATE = .3
TEACHERFORCING_RATIO = 7 # to 1

PRECOMPUTE = False
SEND_ALL_DATA_TO_GPU = False
RESUMETRAINING = False
CHECKPOINT_PATH = 'mostrecent.pt'
EVAL_LOSS_EVERY_NTH_EPOCH = 10

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
  return torch.tensor([[[0. if j != i else 1. for j in range(NUMLABELS)]]], device=device)

def getBatch(sequences):
  xs,xlengths,ys = [],[],[]
  N = len(sequences)
  sampleRandomly = random.randint(0,12412124) % 10 == 0
  k = random.randint(0,N-1)
  for i in range(BATCH_SIZE):
    if sampleRandomly:
      k = random.randint(0,N-1)
    ((x,xlength),y) = sequences[(i+k) % N]
    xs.extend(x)
    xlengths.extend(xlength)
    ys.extend(y)
  ys = torch.tensor(ys,device=device)
  xs = pad_sequence(xs).to(device=device)
  xlengths = torch.tensor(xlengths)
  return (xs,xlengths),ys

class Model(nn.Module):
  def __init__(self, hidden_dim, input_dim, output_dim):
    super(Model, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.encoder = nn.LSTM(input_dim, hidden_dim)
    self.pencoder1 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True, batch_first=True)
    self.pencoder2 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True, batch_first=True)
    self.dropout1 = nn.Dropout(p=DROPOUT_RATE)
    self.dropout2 = nn.Dropout(p=DROPOUT_RATE)
    self.attn = nn.Linear(hidden_dim*2 + output_dim, WINDOWWIDTH//4)
    self.attnCombine = nn.Linear(hidden_dim*2 + output_dim, hidden_dim)
    self.decoder = nn.GRU(hidden_dim, hidden_dim*2, batch_first=True)
    self.out = nn.Linear(hidden_dim*2, output_dim)

  def encode(self, data):
    xs, xlengths = data

    # If batched then xs = (seqlen, WINDOWWIDTH*batches, featuredim)
    packed_padded = pack_padded_sequence(xs, xlengths, enforce_sorted=False)
    packed_padded_out, hidden = self.encoder(packed_padded)
    # Check unpacked_lengths against xlengths to verify correct output ordering
    # unpacked_padded, unpacked_lengths = pad_packed_sequence(packed_padded_hidden[0])

    context_seq = torch.cat(hidden, dim=2) # (1, WINDOWWIDTH * BATCH_SIZE, 2*HIDDEN_DIM)

    context_seq = context_seq.reshape(BATCH_SIZE, WINDOWWIDTH // 2, 4 * HIDDEN_DIM)
    context_seq = self.dropout1(context_seq)
    context_seq, _ = self.pencoder1(context_seq)
    context_seq = context_seq.reshape(BATCH_SIZE, WINDOWWIDTH // 4, 4 * HIDDEN_DIM)
    context_seq = self.dropout2(context_seq)
    context_seq, hidden = self.pencoder2(context_seq)
    hidden = hidden[0] # Take the h vector
    # hidden = (numdirections * layers, batch, hiddensize)
    hidden = hidden.transpose(1,0)
    hidden = hidden.reshape(BATCH_SIZE,2 * HIDDEN_DIM)
    return context_seq, hidden

  def decoderStep(self, input, hidden, encoderOutputs):
    # print('input:',input.shape)
    # print('hidden:',hidden.shape)
    attnWeights = F.softmax(self.attn(torch.cat((input, hidden.unsqueeze(1)), dim=2)), dim=2)
    # print('attnWeights:',attnWeights.shape)
    # print('encoderOutputs:',encoderOutputs.shape)
    attnApplied = torch.bmm(attnWeights, encoderOutputs)
    # print('attnApplied:',attnApplied.shape)
    output = torch.cat((input, attnApplied), dim=2)
    output = self.attnCombine(output)
    output = F.relu(output)
    # print('relu:',output.shape)
    output, hidden = self.decoder(output, hidden.unsqueeze(0))
    hidden = hidden.squeeze(0)
    # print('decoder out:',output.shape)
    # print('decoder hid:',hidden.shape)
    output = F.log_softmax(self.out(output), dim=2)
    # print('log:',output.shape)
    return output, hidden

  def forward(self, xs:torch.Tensor, ys:torch.Tensor=None):
    if ys is not None:
      ys = ys.view(BATCH_SIZE, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)
    # If batched then data = (seqlen, windowlength*batches, featuredim)
    context_seq, hidden = self.encode(xs)
    input = torch.zeros((BATCH_SIZE,1,NUMLABELS), device=device)
    outputs = []
    for i in range(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME):
      output, hidden = self.decoderStep(input, hidden, context_seq)
      if ys is None:
        input = output
      else:
        # TODO could use torch.nn.functional.one_hot
        input = torch.cat([one_hot(ys[j,i]) for j in range(BATCH_SIZE)])
      outputs.append(output)
    output = torch.cat(outputs, dim=1)
    # print(output.shape)
    output = output.view(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME * BATCH_SIZE, NUMLABELS)
    return output

  # # TODO Batchify?
  # def beamDecode(self, data:torch.Tensor):
  #   context_seq, encoder_hidden = self.encode(data)
  #   beams = [] # tuple of (outputs, previous hidden, next input, beam log prob)
  #
  #   # get the initial beam
  #   input = torch.zeros((1,NUMLABELS), device=device)
  #   output, hidden = self.decoderStep(input, encoder_hidden, context_seq)
  #   for i in range(NUMLABELS):
  #     beams.append(([output.view(1,NUMLABELS)], hidden, one_hot(i), float(output[i])))
  #
  #   for i in range(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME - 1):
  #     newBeams = []
  #     for beam in beams:
  #       outputs, hidden, input, beamLogProb = beam
  #       output, hidden = self.decoderStep(input, hidden, context_seq)
  #       for i in range(NUMLABELS):
  #         newBeam = (outputs + [output.view(1,NUMLABELS)], hidden, one_hot(i), beamLogProb + float(output[i]))
  #         newBeams.append(newBeam)
  #     beams = sorted(newBeams, key=lambda x:-x[-1])[:NUMLABELS]
  #
  #   outputs, _, _, _ = beams[0]
  #   return torch.cat(outputs)

# Estimates the loss on the dataset
def evaluate(model, lossFunction, sequences, saveFileName):
  outputs = []
  avgloss = 0
  avgacc = 0
  N = len(sequences)
  for i in range(16):
    xs, ys = getBatch(sequences)

    yhats = model(xs)
    avgloss += float(lossFunction(yhats, ys))

    yhats = yhats.view(BATCH_SIZE, WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME, NUMLABELS)
    yhats = yhats.argmax(dim=2).cpu().numpy()
    ys = ys.view(BATCH_SIZE, WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME).cpu().numpy()
    for j in range(BATCH_SIZE):
      pred = yhats[j]
      exp = ys[j]
      pred = ''.join(['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in pred.tolist()]) + ' '
      exp = ''.join(['.' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in exp.tolist()]) + '\n\n'
      outputs.extend([pred, exp])

    avgacc += ((yhats == ys) & (ys != AllPossibleLabels.index('NOLABEL'))).sum() / (ys == AllPossibleLabels.index('NOLABEL')).sum()

  with open(saveFileName, 'w') as f:
    f.writelines(outputs)

  # # TODO
  # # Beam search is slow so only evaluate it a few times.
  # outputs = []
  # for _ in range(10):
  #   x,y = sequences[random.randint(0,N-1)]
  #   yhat = model.beamDecode(x)
  #   yhat_nobeam = model(x)
  #   avgloss += float(lossFunction(yhat, y))
  #   yhat = yhat.argmax(dim=1)
  #   yhat = yhat.cpu().numpy().tolist()
  #   yhat_nobeam = yhat_nobeam.argmax(dim=1)
  #   yhat_nobeam = yhat_nobeam.cpu().numpy().tolist()
  #   y = y.cpu().numpy().tolist()
  #   if not all([i==AllPossibleLabels.index('NOLABEL') for i in yhat]):
  #     yhat = ''.join(['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in yhat]) + ' '
  #     y = ''.join(['.' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in y]) + '\n\n'
  #     yhat_nobeam = ''.join(['_' if z == AllPossibleLabels.index('NOLABEL') else str(z) for z in yhat_nobeam]) + ' '
  #     outputs.extend(('Beam:   '+yhat+'\n',
  #                     'Greedy: '+yhat_nobeam+'\n',
  #                     'Truth:  '+y+'\n'))
  # with open('beamSample'+saveFileName, 'w') as f:
  #   f.writelines(outputs)

  return avgloss / i, avgacc / i

def checkpoint(epoch, trainloss, testloss, trainacc, testacc, model, optimizer, lossFunction, trainSequences, testSequences):
  model.eval()
  with torch.no_grad():
    avgtrainloss, avgtrainacc = evaluate(model, lossFunction, trainSequences, 'trainOutputs.txt')
    avgtestloss, avgtestacc = evaluate(model, lossFunction, testSequences, 'testOutputs.txt')

    saveData = (model.state_dict(), optimizer.state_dict(), trainloss, testloss, trainacc, testacc)
    if len(trainloss) and avgtrainloss < min(trainloss):
      torch.save(saveData, 'mintrainloss.pt')
    if len(testloss) and avgtestloss < min(testloss):
      torch.save(saveData, 'mintestloss.pt')
    if len(trainacc) and avgtrainacc > max(trainacc):
      torch.save(saveData, 'maxtrainacc.pt')
    if len(testacc) and avgtestacc > max(testacc):
      torch.save(saveData, 'maxtestacc.pt')
    torch.save(saveData, 'mostrecent.pt')

    trainloss.append(avgtrainloss)
    testloss.append(avgtestloss)
    trainacc.append(avgtrainacc)
    testacc.append(avgtestacc)
  model.train()

def train(trainSequences, testSequences, classCounts):
  if SEND_ALL_DATA_TO_GPU:
      print('Sending data to GPU')
      for data in [trainSequences, testSequences]:
        for ((x,xlengths),y) in data:
          for j in range(len(x)):
            x[j] = x[j].to(device)

  print('Training')
  model = Model(HIDDEN_DIM, INPUT_FEATURE_DIM, NUMLABELS)
  model.to(device)
  print(model)

  N = len(trainSequences)
  print('Train set num sequences:',N)
  print('Test set num sequences:',len(testSequences))
  print('Class counts:')
  for label,count in classCounts.items():
    print('\t',label,':',count)

  classWeights = [1/(classCounts[lab]+1) for lab in AllPossibleLabels] # order is important here
  classWeights = torch.tensor(classWeights, device=device) / sum(classWeights)

  lossFunction = nn.NLLLoss(weight=classWeights)
  optimizer = optim.Adam(model.parameters(), lr=.0001)

  trainloss = []
  testloss = []
  trainacc = []
  testacc = []

  if RESUMETRAINING:
    print('Resuming from checkpoint')
    (model_state, optimizer_state, trainloss, testloss, trainacc, testacc) = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad()

  model.train()

  print('Enter training loop')
  for epoch in range(5000):
    start = time.time()
    # for i in tqdm(range(N // BATCH_SIZE)):
    for i in range(N // BATCH_SIZE):
      xs,ys = getBatch(trainSequences)
      if i % TEACHERFORCING_RATIO:
        loss = lossFunction(model(xs,ys), ys)
      else:
        loss = lossFunction(model(xs), ys)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    if epoch % EVAL_LOSS_EVERY_NTH_EPOCH == 0:
      checkpoint(epoch, trainloss, testloss, trainacc, testacc, model, optimizer, lossFunction, trainSequences, testSequences)
      end = time.time()
      print('epoch {} trainloss {:1.5} testloss {:1.5} trainacc {:1.5} testacc {:1.5} time {}'.format(epoch, trainloss[-1], testloss[-1], trainacc[-1], testacc[-1], int(end-start)))
    else:
      end = time.time()
      print('epoch {}                                                                 time {}'.format(epoch, int(end-start)))

  print('Finished training')
  checkpoint(epoch, trainloss, testloss, trainacc, testacc, model, optimizer, lossFunction, trainSequences, testSequences)


# torch.save(dataloader, 'dataloader.pth')
# class MyDataset(Dataset):
#   def __init__(self, size, length):
#     self.len = length
#     self.data = torch.randn(length, size)
#   def __getitem__(self, index):
#     return self.data[index]
#   def __len__(self):
#     return self.len


def getSequenceForInterval(low, high, features, labels, frameNum2Label, allxs, classCounts):
  xs, ys = [], []

  # YOLOboxes, YOLOboxscores, lines, lanescores, vehicles, boxavglaneprobs
  (_, _, lines, lanescores, vehicles, boxavglaneprobs) = features

  # Only look at data from this interval of frames
  vehicles = vehicles[low:high]
  boxavglaneprobs = boxavglaneprobs[low:high]
  lanescores = lanescores[low:high]
  lines = lines[low:high]

  # remap objId's to sensible range (1-N) where N is on average about 10.
  objIds = set()
  for vehicles_ in vehicles:
    for vehicle in vehicles_:
      objIds.add(vehicle.objId)
  objIds = sorted(objIds)

  # for the jth frame in the interval
  for j, (vehicles_, centroidProbs, laneScores_, lines_) in enumerate(zip(vehicles, boxavglaneprobs, lanescores, lines)):
    laneScores_ = [float(score.cpu().numpy()) for score in laneScores_]
    leftLane = lines_[0]
    rightLane = lines_[1]

    centroidProbsLeft = centroidProbs[0]
    centroidProbsRight = centroidProbs[1]

    features_xs = []
    # descending sort by percent life completed
    prioritized = sorted(list(zip(vehicles_, centroidProbsLeft, centroidProbsRight)),
                         key=lambda tup:-tup[0].percentLifeComplete)
    for vehicle, centroidProbLeft, centroidProbRight in prioritized:
      # Put object id into sensible range
      vehicle.objId = objIds.index(vehicle.objId)
      tens = torch.tensor([[*vehicle.unpack(), centroidProbLeft, centroidProbRight, *laneScores_, *leftLane, *rightLane]])
      features_xs.append(tens)
    allxs.extend(features_xs)

    # there may be lane changes even if there are no boxes in the video
    if len(features_xs) == 0:
      tens = torch.tensor([[0]*12 + [*laneScores_, *leftLane, *rightLane]])  # make sure there is always an input tensor
      features_xs.append(tens)

    features_xs = torch.cat(features_xs)
    xs.append(features_xs)
    if (low+j) % PREDICT_EVERY_NTH_FRAME == 0:
      if (low+j)//PREDICT_EVERY_NTH_FRAME in frameNum2Label:
        label = frameNum2Label[(low+j)//PREDICT_EVERY_NTH_FRAME]
        classCounts[label] += 1
        ys.append(labels2Tensor[label])
      else:
        classCounts['NOLABEL'] += 1
        ys.append(labels2Tensor['NOLABEL'])

  xlengths = list(map(len, xs))
  return ((xs, xlengths),ys)

vehiclesKept = set()
vehiclesRemoved = set()
class Vehicle:
  def __init__(self, vehicle):
    objId, x1, y1, x2, y2 = vehicle
    self.objId = objId
    self.point = (x1,y1,x2,y2)
    self.maxAge = None
    self.ageAtFrame = None
    self.percentLifeComplete = None
    self.avgSignedDistToLeftLane = None
    self.avgSignedDistToRightLane = None
  def unpack(self):
    assert(self.objId is not None)
    assert(self.maxAge is not None)
    assert(self.ageAtFrame is not None)
    assert(self.percentLifeComplete is not None)
    assert(self.avgSignedDistToLeftLane is not None)
    assert(self.avgSignedDistToRightLane is not None)
    return (self.objId,
            *self.point,
            self.maxAge,
            self.ageAtFrame,
            self.avgSignedDistToLeftLane,
            self.avgSignedDistToRightLane,
            self.percentLifeComplete)

def getSequencesForFiles(files, allxs, classCounts):
  sequences = []
  for fileI, file in enumerate(files):
    labelsFilePath = file.replace('features/', 'labels/').replace('_m0.pkl', '_labels.txt')
    if not os.path.isfile(labelsFilePath):
      continue
    print('\t processing '+str(fileI)+'th file')

    # Get features
    features = None
    with open(file, 'rb') as featuresfile:
      features = list(pickle.load(featuresfile))

    boxes, boxscores, lines, lanescores, vehicles, boxavglaneprobs = features

    # Do a few things here;
    # Add a feature to describe the age in number frames of the box.
    # Filter boxes that do not exist in the video for long enough to matter.
    # Add a feature to describb how far and on which side of a lane the vehicle is.
    # This loop will also convert all the vehicles into Vehicle objects which
    # are infinitely easier to work with than raw tuples. This feature engineering
    # could have been done in Classifier.py but, well, we didn't do it and so we'll
    # do it here instead of recomputing features.
    objFirstFrame = dict()
    maxAge = dict()
    for i in range(len(vehicles)):
      vehiclesInFramei = vehicles[i]
      for j,vehicleTuple in enumerate(vehiclesInFramei):
        v = Vehicle(vehicleTuple)
        if v.objId not in objFirstFrame:
          objFirstFrame[v.objId] = i
        # Add age at frame feature
        v.ageAtFrame = i - objFirstFrame[v.objId]
        maxAge[v.objId] = max(maxAge.get(v.objId, 0), v.ageAtFrame)
        vehiclesInFramei[j] = v
    # Add max age feature, filter the boxes, and calculate avg lane distances
    for i in range(len(vehicles)):
      if len(lines[i]) != 2:
        lines[i] = [[],[]]
      leftLane, rightLane = lines[i]
      if len(leftLane) == 0: leftLane = [(0,0,0,0)]
      if len(rightLane) == 0: rightLane = [(0,0,0,0)]
      vehiclesInFramei = vehicles[i]
      newVehiclesInFramei = []
      for v in vehiclesInFramei:
        if maxAge[v.objId] >= 15:
          vehiclesKept.add(v.objId)
          # Add max age feature
          v.maxAge = maxAge[v.objId]
          v.percentLifeComplete = v.ageAtFrame / v.maxAge
          # Add signed distance feature
          x1,y1,x2,y2 = v.point
          midX = (x1 + x2) / 2
          v.avgSignedDistToLeftLane = (np.array([(x1+x2)/2 for (x1,y1,x2,y2) in leftLane]) - midX).mean()
          v.avgSignedDistToRightLane = (np.array([(x1+x2)/2 for (x1,y1,x2,y2) in rightLane]) - midX).mean()
          newVehiclesInFramei.append(v)
        else:
          vehiclesRemoved.add(v.objId)
      vehicles[i] = newVehiclesInFramei

    # Take every 17th point on the line
    for i in range(len(lines)):
      if len(lines[i]) != 2:
        lines[i] = [[],[]]
      left,right = lines[i]
      left = left[::17][:10]
      right = right[::17][:10]
      if len(left) < 10:
        left += [(0,0,0,0)] * (10 - len(left))
      if len(right) < 10:
        right += [(0,0,0,0)] * (10 - len(right))
      left = [x1 for (x1,y1,x2,y2) in left]
      right = [x1 for (x1,y1,x2,y2) in right]
      lines[i] = (left,right)

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
    for i in range(0, 30*60*5 - WINDOWWIDTH, WINDOWSTEP):
      [low, high] = [i, i + WINDOWWIDTH]
      # Search for a label that is in the interval.
      # If such a label does not exist, then we will not add the sequence to the dataset.
      for label, frameNum in labels:
        if low <= frameNum <= high:
          data = getSequenceForInterval(low, high, features, labels, frameNum2Label, allxs, classCounts)
          sequences.append(data)
          # There may be more than one label in this interval.
          # If we ran this loop twice in this interval then we would append the same exact (xs,ys) to sequences
          break
  return sequences

if __name__ == '__main__':
  print('Loading data.')

  if PRECOMPUTE:
    with torch.no_grad():
      # Get paths to precomputed features
      filepaths = []
      for (dirpath, dirnames, filenames) in os.walk('features'):
        filepaths.extend(dirpath + '/' + f for f in filenames)
      random.shuffle(filepaths)

      trainSize = int(len(filepaths)*.85)
      trainSet = filepaths[:trainSize]
      testSet = filepaths[trainSize:]

      print('Train set has size:', len(trainSet))
      print('Test set has size:', len(testSet))

      classCounts = {label:0 for label in AllPossibleLabels}
      allxs = [] # for computing mean and std

      # Convert the data into tensors
      trainSequences = getSequencesForFiles(trainSet, allxs, classCounts)
      testSequences = getSequencesForFiles(testSet, allxs, classCounts)

      print('Class counts for data:',classCounts)
      print('Vehicles skipped:',len(vehiclesRemoved),
            'Vehicles kept:', len(vehiclesKept))

      print('Calculating statistics')
      bigboy = torch.cat(allxs)
      mean = bigboy.mean(dim=0).clone()
      std = bigboy.std(dim=0).clone()
      print('Shape of full data matrix:',bigboy.shape)
      del bigboy
      del allxs

      print('Augmenting data')
      extraDataTrain = []
      for i,((x,xlengths),y) in enumerate(trainSequences[::3]):
        newseqReflH = []
        # newseqRandShift = []
        newseqNoise = []
        for j in range(len(x)):
          # reflect everything left to right
          tens = x[j].clone()
          tens[0,1] *= -1
          tens[0,1] += 720
          tens[0,3] *= -1
          tens[0,3] += 720
          tens[0,-20:] *= -1
          tens[0,-20:] += 720
          newseqReflH.append(tens)
          tens = x[j].clone()
          tens[0,1:9] += torch.randn_like(tens[0,1:9]) * 40
          tens[0,-20:] += torch.randn_like(tens[0,-20:]) * 40
          # newseqRandShift.append(tens)
          # tens = x[j].clone()
          # tens += torch.randn_like(tens)
          newseqNoise.append(tens)
        extraDataTrain.append(((newseqReflH,xlengths),y))
        # extraDataTrain.append(((newseqRandShift,xlengths),y))
        extraDataTrain.append(((newseqNoise,xlengths),y))
      trainSequences.extend(extraDataTrain)

      print('Standardizing data')
      # TODO We should record the mean and std of the dataset our model was trained on
      # In production we need to try and make the input data lie in the same distribution
      for data in [trainSequences, testSequences]:
        for ((x,xlengths),y) in data:
          for j in range(len(x)):
            x[j] -= mean
            x[j] /= std

      print('Writing tensors')
      torch.save((trainSequences, testSequences, classCounts), 'tensors.pkl')
      print('Wrote tensors pickle. Exiting.')
  else:
    (trainSequences, testSequences, classCounts) = torch.load('tensors.pkl')
    train(trainSequences, testSequences, classCounts)
