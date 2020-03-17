import os
import pickle
with open('predictions.pkl', 'rb') as file:
    sequences = pickle.load(file)
    labels=[]
    i = 0
    j = 0
    h = 0
    for seq in sequences:
        if i<=4:
            for val in seq:
                label = val
                #print(val.data.cpu().numpy() == '8')
                if label != 8:
                    if label == 0:
                        label = "rightTO"
                    elif label == 1:
                        label = "lcRel"
                    elif label == 2:
                        label = "cutin"
                    elif label == 3:
                        label = "cutout"
                    elif label == 4:
                        label = "evtEnd"
                    elif label == 5:
                        label = "cutin"
                    elif label == 6:
                        label = "objTurnOff"
                    elif label == 7:
                        label = "barrier"
                    timestamp = j/30
                    tup = (label, timestamp)
                    labels.append(tup)
                    h+=1
                #print(val.data.cpu().numpy())
                j += 1
            i+=1
        #print (seq[1])
    #print(sequences)
    print(j)
    print(h)
    print(labels)