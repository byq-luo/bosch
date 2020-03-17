import matplotlib.pyplot as plt
import numpy as np
import pickle

def ewm(xs,m):
  ret = np.zeros_like(xs)
  ret[0] = xs[0]
  for i,x in enumerate(xs[1:]):
    ret[i+1] = (ret[i] * m + x * (1-m))
  return ret

SMOOTH = .99

with open('trainlossNoise.pkl','rb') as f:
  trainloss = np.array(pickle.load(f))
with open('testlossNoise.pkl','rb') as f:
  testloss = np.array(pickle.load(f))

smoothtrainloss = ewm(trainloss,SMOOTH)
smoothtestloss = ewm(testloss,SMOOTH)

x1 = np.linspace(0, 1, len(trainloss))
x2 = np.linspace(0, 1, len(testloss))

plt.plot(x1,smoothtrainloss)
plt.plot(x2,smoothtestloss,color=(0,0,0))
plt.show()

