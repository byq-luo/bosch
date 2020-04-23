import os, time, random, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from Dataset import Dataset

PREDICT_EVERY_NTH_FRAME = 30
WINDOWWIDTH = 30*30
AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'BLANK']
NUMLABELS = len(AllPossibleLabels)
BATCH_SIZE = 64
RESUME_TRAINING = True
LOAD_CHECKPOINT_PATH = 'mostrecent.pt'

DMODEL = 32
DFF = 32
DROPOUT_P = .1
LAYERS = 1
HEADS = 1
INP_DIM = 63
LR = .0003

#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(29809284)
device = torch.device('cuda')
devcount = torch.cuda.device_count()
print('Device count:',devcount)
torch.cuda.set_device(devcount-1)
print('Current device:',torch.cuda.current_device())
print('Device name:',torch.cuda.get_device_name(devcount-1))

# Start of sequence decoder input
SOS = torch.full((1,), len(AllPossibleLabels), dtype=torch.long, device=device)

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

#    return (self.objId, #1
#            self.maxAge, #2
#            self.ageAtFrame, #3
#            self.width, #4
#            self.height, #5
#            self.area, #6
#            self.percentLifeComplete, #7
#            self.inHostLane, #8
#            x1  # 9
#            y1  # 10
#            x2  # 11
#            y2  # 12
#            self.avgSignedDistToLeftLane, # 13
#            self.avgSignedDistToRightLane, # 14
#            self.centroidProbLeft, # 15
#            self.centroidProbRight, # 16
#            self.inLeftLane, # 17
#            self.inRightLane) # 18
#           lanescores # 19,20,21,22
#           left # 23-42
#           right # 43-62

def create_masks(sz):
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

def augment(t):
  tens = t.clone()
  i = random.randint(0,2)
  if i == 0:
    tens[:,9], tens[:,11] = -tens[:,11], -tens[:,9] # x1,x2
    tens[:,13], tens[:,14] = -tens[:,14], -tens[:,13] # avg signed dist
    tens[:,15], tens[:,16] = tens[:,16], tens[:,15] # centroid prob
    tens[:,17], tens[:,18] = tens[:,18], tens[:,17] # in left/right lane
    tens[:,19], tens[:,20], tens[:,21], tens[:,22] = tens[:,22], tens[:,21], tens[:,20], tens[:,19] # lane scores
    tens[:,-40:-20], tens[:,-20:] = -tens[:,-20:], -tens[:,-40:-20] # lane coords
  elif i==1:
    tens[:,9], tens[:,11] = -tens[:,11], -tens[:,9] # x1,x2
    tens[:,13], tens[:,14] = -tens[:,14], -tens[:,13] # avg signed dist
    tens[:,15], tens[:,16] = tens[:,16], tens[:,15] # centroid prob
    tens[:,17], tens[:,18] = tens[:,18], tens[:,17] # in left/right lane
    tens[:,19], tens[:,20], tens[:,21], tens[:,22] = tens[:,22], tens[:,21], tens[:,20], tens[:,19] # lane scores
    tens[:,-40:-20], tens[:,-20:] = -tens[:,-20:], -tens[:,-40:-20]
    tens += torch.randn_like(tens) * .1
  else:
    tens += torch.randn_like(tens) * .1
  return tens

def getBatch(dataset, aug=True, size=BATCH_SIZE):
  batch_xs,batch_xlengths,batch_ys = [],[],[]
  N = len(dataset)
  for i in range(size):
    ((xs,xlengths),ys) = dataset[random.randint(0,N-1)]
    if aug and i % 3 == 0:
      xs = [augment(x) for x in xs]
    batch_xs.extend(xs)
    batch_xlengths.extend(xlengths)
    batch_ys.extend(ys)
  batch_xs = pad_sequence(batch_xs).to(device=device)
  batch_xlengths = torch.tensor(batch_xlengths, device=device)
  batch_ys = torch.tensor(batch_ys, device=device)
  return (batch_xs,batch_xlengths),batch_ys

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=-1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0],:].detach()
        return self.dropout(x)

def evaluate(model, sequences, saveFileName):
  start = time.time()

  outputs = []
  avgacc1 = 0
  avgacc2 = 0
  avgiou = 0

  SAMPLES = 16

  for i in range(SAMPLES):
    xs, trg = getBatch(sequences, aug=False)
    trg = trg.view(BATCH_SIZE,WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME).cpu().numpy()
    probs = model.translate_sentence(xs)
    yhats = probs.argmax(dim=2).cpu().numpy()
    for j in range(BATCH_SIZE):
      pred = ['_' if z == AllPossibleLabels.index('BLANK') else str(z) for z in yhats[j].tolist()]
      exp = ['.' if z == AllPossibleLabels.index('BLANK') else str(z) for z in trg[j].tolist()]
      a = set(yhats[j].tolist())
      b = set(trg[j].tolist())
      avgiou += len(a&b)/len(a|b) / BATCH_SIZE

      outputs.append(''.join(pred) + ' ' + ''.join(exp) + '\n\n')
    numlabels = (trg != AllPossibleLabels.index('BLANK')).sum()
    if numlabels > 0:
      avgacc1 += ((yhats == trg) & (trg != AllPossibleLabels.index('BLANK'))).sum() / numlabels
    avgacc2 += (yhats == trg).sum() / (BATCH_SIZE * (WINDOWWIDTH // PREDICT_EVERY_NTH_FRAME))

  end = time.time()
  outputs = ['acc:'+str(avgacc1/SAMPLES)+' iou:'+str(avgiou/SAMPLES)+' evaltime:'+str(int(end-start))+'\n\n'] + outputs

  return avgacc1 / SAMPLES, avgacc2 / SAMPLES, avgiou / SAMPLES, outputs

def checkpoint(trainloss, losses, model, optimizer, trainData, testData):
  model.eval()
  with torch.no_grad():
    avgtrainacc, avgtrainacc2, avgtrainiou, trainOuts = evaluate(model, trainData, 'trainOutputs.txt')
    avgtestacc, avgtestacc2, avgtestiou, testOuts = evaluate(model, testData, 'testOutputs.txt')
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


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.myenc = nn.GRU(src_vocab, d_model)
        self.embed = nn.Embedding(trg_vocab + 1, d_model) # plus 1 for the sos token
        self.trans = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=N, num_decoder_layers=N, dim_feedforward=DFF, dropout=dropout)
        self.out = nn.Linear(d_model, NUMLABELS)
        self.pos_enc1 = PositionalEncoding(d_model, dropout,max_len=WINDOWWIDTH)
        self.pos_enc2 = PositionalEncoding(d_model, dropout,max_len=WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)
        self.d_model = d_model

    # teacher forcing
    def forward(self, src, trg, trg_mask):

        xs, xlengths = src
        batch_size = xs.shape[1] // WINDOWWIDTH
        packed_padded = pack_padded_sequence(xs, xlengths, enforce_sorted=False)
        packed_padded_out, hidden = self.myenc(packed_padded)
        seq = hidden.view(batch_size, WINDOWWIDTH, self.d_model)
        seq = seq.transpose(0,1) # TxBxE
        seq = self.pos_enc1(seq)

        embed = self.embed(trg) * math.sqrt(self.d_model)
        embed = embed.transpose(0,1) # TxBxE
        embed = self.pos_enc2(embed)
        output = self.trans(seq, embed, tgt_mask=trg_mask)
        output = output.transpose(0,1)
        return self.out(output)

    # no teacher forcing
    def _model_decode(self, trg, enc_output): # trg = BxT
        # Removing clone here would be an error. the trg tensor is modified in place outside of this function.
        trg = self.embed(trg.clone().detach()) * math.sqrt(self.d_model)
        trg = trg.transpose(0,1) # TxB
        trg = self.pos_enc2(trg)
        dec_output = self.trans.decoder(trg, enc_output)
        return self.out(dec_output.transpose(0,1))

    def _get_init_state(self, src):
        xs, xlengths = src
        batch_size = xs.shape[1] // WINDOWWIDTH
        packed_padded = pack_padded_sequence(xs, xlengths, enforce_sorted=False)
        packed_padded_out, hidden = self.myenc(packed_padded)

        seq = hidden.view(batch_size, WINDOWWIDTH, self.d_model)
        seq = seq.transpose(0,1)
        src_seq = self.pos_enc1(seq)
        enc_output = self.trans.encoder(src_seq)

        gen_seq = torch.zeros((batch_size, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME), dtype=torch.long, device=device)
        sos = SOS.repeat(batch_size, 1)
        gen_seq[:,0] = sos[:,0]

        dec_output = self._model_decode(gen_seq[:,:1], enc_output)
        dec_output = dec_output[:, -1:, :] # take the final timestep
        return enc_output, dec_output, gen_seq

    def translate_sentence(self, src_seq):
        enc_output, dec_output, gen_seq = self._get_init_state(src_seq)
        outputs = [dec_output]

        for step in range(1, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME):
            best_idx = dec_output.argmax(dim=2)
            gen_seq[:, step] = best_idx[:,0]
            dec_output = self._model_decode(gen_seq[:,:step+1], enc_output)
            dec_output = dec_output[:, -1:, :] # take the final timestep
            outputs.append(dec_output)

        return torch.cat(outputs,dim=1) # BxW/PxL

if __name__ == '__main__':
  trg_mask = create_masks(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME).to(device)
  sos = SOS.repeat(BATCH_SIZE,1)

  trainData = Dataset(loadPath='../trainDataset.pkl')
  testData = Dataset(loadPath='../testDataset.pkl')

  (_,_,classCounts) = trainData.getStats()
  print('loaded data')

  N = len(trainData)
  print('Train set num data:',N)
  print('Test set num data:',len(testData))
  print('Class counts:')
  for label,count in classCounts.items():
    print('\t',label,':',count)
  classWeights = [1/(classCounts[lab]+1) for lab in range(NUMLABELS)] # order is important here
  classWeights = torch.tensor(classWeights, device=device) / sum(classWeights)

  model = Transformer(INP_DIM, NUMLABELS, DMODEL, LAYERS, HEADS, DROPOUT_P)
  model.to(device)
  losses = Losses()

  optimizer = optim.Adam(model.parameters(), lr=LR)
  #sched = optim.lr_scheduler.ExponentialLR(optimizer, .981)

  if RESUME_TRAINING:
    print('Resuming from checkpoint')
    (model_state, optimizer_state) = torch.load(LOAD_CHECKPOINT_PATH,map_location=device)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad()

  model.train()
  print(model)

  losses = Losses()

  avgloss = -math.log(1/NUMLABELS)
  prev = time.time()
  i = 0
  while True:

    src, trg = getBatch(trainData)
    trg = trg.view(BATCH_SIZE,WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)
    ys = trg.view(-1)
    trg = torch.cat([sos,trg[:,:-1]], dim=1)

    if i % 5:
      preds = model(src, trg, trg_mask)
    else:
      preds = model.translate_sentence(src)

    optimizer.zero_grad()
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, weight=classWeights)
    loss.backward()
    optimizer.step()

    avgloss += .05 * (float(loss) - avgloss)
    i += 1
    if i % 5 == 0:
      print(avgloss)

    now = time.time()
    if now - prev > 60:
      testacc, testiou = checkpoint(avgloss, losses, model, optimizer, trainData, testData)
      print('trainloss {:1.5} testacc {:1.5} testiou {:1.5}'.format(avgloss, testacc, testiou), flush=True)
      prev = time.time()
      #sched.step()
