import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Losses:
  def __init__(self):
      self.trainLoss = []
      self.testLoss = []
      self.trainAcc = []
      self.testAcc = []
      self.trainAcc2 = []
      self.testAcc2 = []


def ewm(xs,m):
  ret = np.zeros_like(xs)
  ret[0] = xs[0]
  for i,x in enumerate(xs[1:]):
    ret[i+1] = (ret[i] * m + x * (1-m))
  return ret

SMOOTH = .9

(_, _, losses) = torch.load('mostrecent.pt')

print('Number of points:',len(losses.trainLoss))

smoothtrainloss = ewm(losses.trainLoss,SMOOTH)
smoothtestloss = ewm(losses.testLoss,SMOOTH)
smoothtrainacc = (40-ewm(losses.trainAcc,SMOOTH))/40
smoothtestacc = (40 - ewm(losses.testAcc,SMOOTH)) / 40

x1 = np.linspace(0, 1, len(losses.trainLoss))
x2 = np.linspace(0, 1, len(losses.testLoss))
x3 = np.linspace(0, 1, len(losses.trainAcc))
x4 = np.linspace(0, 1, len(losses.testAcc))

# plt.plot(x1,smoothtrainloss,color=(0,0,0))
# plt.plot(x2,smoothtestloss,color=(1,0,0))
plt.plot(x3,smoothtrainacc,color=(0,1,0))
plt.plot(x4,smoothtestacc,color=(0,0,1))
plt.legend
plt.show()

