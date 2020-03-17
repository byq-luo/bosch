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

device = torch.device('cuda')

def clamp(l, h, x):
  if x < l: return l
  if x > h: return h
  return x

class Model(nn.Module):
  def __init__(self, hidden_dim, input_dim):
    super(Model, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.encoder = nn.LSTM(input_dim, hidden_dim)
    self.trans = nn.Linear(hidden_dim, input_dim)
    self.decoder = nn.LSTM(input_dim, input_dim)
    self.startToken = -torch.ones((1,240,self.input_dim), device=device) / self.input_dim

  def forward(self, data):
    _, context_seq = self.encoder(data)
    revdata = torch.flip(data,dims=(0,))
    h = self.trans(context_seq[0])
    c = self.trans(context_seq[1])
    revdata = torch.cat([self.startToken, revdata], dim=0)
    seq, _ = self.decoder(revdata, (h,c))
    seq = seq[:-1,:,:]
    return seq


def evaluate(model, losses, loss_function, sequences):
  average_loss = 0
  for i,(x,_) in enumerate(sequences):
    y = torch.flip(x,dims=(0,))
    yh = model(x)
    loss = float(loss_function(yh, y))
    if i == 0:
      r = random.randint(0,239)
      print('y:',y[0,r,:].cpu().numpy().tolist())
      print('yh:',yh[0,r,:].cpu().numpy().tolist())
    losses.append(loss)
    average_loss += loss
  print(average_loss / len(sequences))


def checkpoint(epoch, trainloss, testloss, model, loss_function, trainSequences, testSequences):
  model.eval()
  with torch.no_grad():
    print('Saving model at epoch:', epoch)
    torch.save(model.state_dict(), 'modelAUTO.pt')
    print('-'*20,'TRAINING','-'*20)
    evaluate(model, trainloss, loss_function, trainSequences)
    print('-'*20,'TESTING','-'*20)
    evaluate(model, testloss, loss_function, testSequences)
    with open('trainlossAUTO.pkl', 'wb') as file:
      pickle.dump(trainloss, file)
    with open('testlossAUTO.pkl', 'wb') as file:
      pickle.dump(testloss, file)
    print('Saved', epoch)
  model.train()


def train(trainSequences, testSequences):
  print('Training')
  model = Model(128, 17)
  model.to(device)
  model.train()
  loss_function = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=.05)
  print('Enter training loop')
  ema_loss = None
  trainlosses = []
  testlosses = []
  for epoch in range(5000):
    for i in range(len(trainSequences)):
      k = random.randint(0,len(trainSequences)-1)
      x,_ = trainSequences[k]
      y = torch.flip(x,dims=(0,))
      optimizer.zero_grad()
      loss = loss_function(model(x), y)
      loss.backward()
      optimizer.step()
    if epoch % 4 == 0:
      checkpoint(epoch, trainlosses, testlosses, model, loss_function, trainSequences, testSequences)
  print('Finished training')
  checkpoint(epoch, losses, model, loss_function, trainSequences, testSequences)


if __name__ == '__main__':
  print('Loading data.')
  with open('tensors.pkl', 'rb') as file:
    (trainSequences, testSequences) = pickle.load(file)
  train(trainSequences, testSequences)
