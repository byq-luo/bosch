import matplotlib.pyplot as plt
import pickle
def smooth(xs,m):
  ret = [xs[0]]
  for x in xs[1:]:
    ret.append(ret[-1] * m + x * (1-m))
  return ret
with open('losses.pkl','rb') as f:
  losses = pickle.load(f)
smoothloss = smooth(losses,.999)
plt.plot(smoothloss)
plt.show()
