import os, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import copy

from torch.autograd import Variable

import numpy as np

import math
from Dataset import Dataset

torch.manual_seed(29809284)
device = torch.device('cuda')
devcount = torch.cuda.device_count()
print('Device count:',devcount)
torch.cuda.set_device(devcount-1)
print('Current device:',torch.cuda.current_device())
print('Device name:',torch.cuda.get_device_name(devcount-1))

PREDICT_EVERY_NTH_FRAME = 30
WINDOWWIDTH = 30*30
AllPossibleLabels = ['rightTO', 'lcRel', 'cutin', 'cutout', 'evtEnd', 'BLANK']
NUMLABELS = len(AllPossibleLabels)
BATCH_SIZE = 1
LOAD_CHECKPOINT_PATH = 'mostrecent.pt'

DMODEL = 64
DFF = 128
DROPOUT_P = 0
LAYERS = 6
HEADS = 4
INP_DIM = 63
LR = .0003

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

class Translator(nn.Module):
    def __init__(self, model, beam_size, max_seq_len):
        super(Translator, self).__init__()
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.model = model
        self.model.eval()
        self.register_buffer('init_seq', torch.LongTensor([[AllPossibleLabels.index('BLANK')]]))
        self.register_buffer('blank_seqs', torch.full((beam_size, max_seq_len), AllPossibleLabels.index('BLANK'), dtype=torch.long))
        self.d_model = self.model.d_model
        self.pos_enc1 = PositionalEncoding(self.d_model, 0, max_len=WINDOWWIDTH)
        self.pos_enc2 = PositionalEncoding(self.d_model, 0, max_len=WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME-1)
    def _model_decode(self, trg, hi, enc_output, trg_mask):
        # embed?
        # positional enc inp
        print('intrg:',trg.shape)
        print('inhi:',hi.shape)
        hi = self.model.embed(hi[:,:trg.shape[1]])
        hi = hi.transpose(0,1)
        hi = self.pos_enc2(hi * math.sqrt(self.d_model))

        trg = self.model.embed(trg)
        trg = trg.transpose(0,1)
        trg = self.pos_enc2(trg * math.sqrt(self.d_model))
        print(trg.shape)
        print(enc_output.shape)
        print(hi.shape)
        dec_output = self.model.trans.decoder(hi, enc_output, tgt_mask=trg_mask[:trg.shape[0],:trg.shape[0]])
        return F.log_softmax(self.model.out(dec_output.transpose(0,1).contiguous()), dim=-1)
    def _get_init_state(self, src, hi, trg_mask):
        beam_size = self.beam_size

        xs, xlengths = src
        batch_size = xs.shape[1] // WINDOWWIDTH
        print(batch_size)
        packed_padded = pack_padded_sequence(xs, xlengths, enforce_sorted=False)
        packed_padded_out, hidden = self.model.myenc(packed_padded)

        seq = hidden.view(batch_size, WINDOWWIDTH, self.d_model)
        seq = seq.transpose(0,1)
        print('hiiiiiiiiiiiiiiiiiii:',seq.shape)
        src_seq = self.pos_enc1(seq * math.sqrt(self.d_model))
        enc_output = self.model.trans.encoder(src_seq)
        enc_output = enc_output.repeat(1,self.beam_size,1)
        print('byeiiiiiiiiiiiiiiiiiii:',enc_output.shape)

        # positional enc initseq
        dec_output = self._model_decode(self.init_seq, hi, enc_output[:,:1,:], trg_mask)
        print('decout:',dec_output)
        print('decout:',dec_output.shape)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)
        scores = best_k_probs.view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        return enc_output, gen_seq, scores
    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        print('hai',dec_output.shape)
        assert len(scores.size()) == 1
        beam_size = self.beam_size
        # Get k candidates for each beam, k^2 candidates in total.
        print('decout:',dec_output.shape)
        dec_output = dec_output[:, -1, :]
        best_k2_probs, best_k2_idx = dec_output.topk(beam_size)
        print('decout:',dec_output.shape)
        print(best_k2_probs.shape)
        # Include the previous scores.
        scores = best_k2_probs.view(beam_size, -1) + scores.view(beam_size, 1)
        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx
        return gen_seq, scores
    def translate_sentence(self, src_seq, hi, trg_mask):
        with torch.no_grad():
            enc_output, gen_seq, scores = self._get_init_state(src_seq, hi, trg_mask)
            ans_idx = 0   # default
            for step in range(2, self.max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], hi, enc_output, trg_mask)
                print(gen_seq)
                print(scores)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
        return gen_seq[ans_idx]

def augment(t, i):
  tens = t.clone()
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

def getBatch(i, dataset, aug=True, size=BATCH_SIZE):
  batch_xs,batch_xlengths,batch_ys = [],[],[]
  N = len(dataset)
  for j in range(i,i+size):
    ((xs,xlengths),ys) = dataset[j%N]
    batch_xs.extend(xs)
    batch_xlengths.extend(xlengths)
    batch_ys.extend(ys)
  batch_xs = pad_sequence(batch_xs).to(device=device)
  batch_xlengths = torch.tensor(batch_xlengths, device=device)
  batch_ys = torch.tensor(batch_ys, device=device)
  return (batch_xs,batch_xlengths),batch_ys

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME-1):
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
        return self.dropout(x + self.pe[:x.shape[0]])

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.myenc = nn.GRU(src_vocab, d_model)
        self.embed = nn.Embedding(trg_vocab, d_model)
        self.trans = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=N, num_decoder_layers=N, dim_feedforward=DFF, dropout=dropout)
        self.out = nn.Linear(d_model, NUMLABELS)
        self.pos_enc1 = PositionalEncoding(d_model, dropout,max_len=WINDOWWIDTH)
        self.pos_enc2 = PositionalEncoding(d_model, dropout,max_len=WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME-1)
        self.d_model = d_model
    def forward(self, src, trg, trg_mask):
        xs, xlengths = src
        batch_size = xs.shape[1] // WINDOWWIDTH
        packed_padded = pack_padded_sequence(xs, xlengths, enforce_sorted=False)
        packed_padded_out, hidden = self.myenc(packed_padded)
        seq = hidden.view(batch_size, WINDOWWIDTH, self.d_model)
        seq = seq.transpose(0,1)
        seq = self.pos_enc1(seq * math.sqrt(self.d_model))
        trg = self.embed(trg)
        trg = trg.transpose(0,1)
        trg = self.pos_enc2(trg * math.sqrt(self.d_model))
        output = self.trans(seq, trg, tgt_mask=trg_mask)
        return self.out(output.transpose(0,1).contiguous())

if __name__ == '__main__':
  trg_mask = create_masks(WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME-1).to(device)
  print(trg_mask)
  testData = Dataset(loadPath='testDataset.pkl')
  model = Transformer(INP_DIM, NUMLABELS, DMODEL, LAYERS, HEADS, 0)
  model.to(device)
  (model_state, optimizer_state) = torch.load(LOAD_CHECKPOINT_PATH,map_location=device)
  model.load_state_dict(model_state)
  model.eval()
  translator = Translator(model, 1, WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME) 
  translator.to(device)
  translator.eval()
  outputs = [] 
  for i in range(0,len(testData),BATCH_SIZE):
    xs, ys = getBatch(i, testData, aug=False, size=1)
    hi = ys.view(BATCH_SIZE,WINDOWWIDTH//PREDICT_EVERY_NTH_FRAME)[:, :-1]
    mhats = model(xs,hi,trg_mask).argmax(dim=2)[0]
    yhats = translator.translate_sentence(xs,hi, trg_mask)
    for j in range(BATCH_SIZE):
      pred = ['_' if z == AllPossibleLabels.index('BLANK') else str(z) for z in yhats.tolist()]
      pred2 = ['_' if z == AllPossibleLabels.index('BLANK') else str(z) for z in mhats.tolist()]
      exp = ['.' if z == AllPossibleLabels.index('BLANK') else str(z) for z in ys.tolist()]
      a = set(yhats.tolist())
      b = set(ys.tolist())
      out  = ''.join(pred) + ' ' + ''.join(exp) + ' ' + ''.join(pred2) + '\n\n'
      print(out)
      outputs.append(out)

  with open('outputs.txt','w') as f:
    f.writelines(outputs)
