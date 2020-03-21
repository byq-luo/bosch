import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

def ewm(xs,m):
  ret = np.zeros_like(xs)
  ret[0] = xs[0]
  for i,x in enumerate(xs[1:]):
    ret[i+1] = (ret[i] * m + x * (1-m))
  return ret

SMOOTH = .9

(_, _, trainloss, testloss, trainacc, testacc) = torch.load('mostrecent.pt')

print('Number of points:',len(trainloss))

smoothtrainloss = ewm(trainloss,SMOOTH)
smoothtestloss = ewm(testloss,SMOOTH)
smoothtrainacc = (40-ewm(trainacc,SMOOTH))/40
smoothtestacc = (40 - ewm(testacc,SMOOTH)) / 40

x1 = np.linspace(0, 1, len(trainloss))
x2 = np.linspace(0, 1, len(testloss))
x3 = np.linspace(0, 1, len(trainacc))
x4 = np.linspace(0, 1, len(testacc))

plt.plot(x1,smoothtrainloss,color=(0,0,0))
plt.plot(x2,smoothtestloss,color=(1,0,0))
plt.plot(x3,smoothtrainacc,color=(0,1,0))
plt.plot(x4,smoothtestacc,color=(0,0,1))
plt.show()

