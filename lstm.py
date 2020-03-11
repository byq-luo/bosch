import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
import numpy as np
import pickle
import os

class mylstm(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_size):
        super(mylstm, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.hidden = None

    def forward(self, inputs):
        # add the batch dimension
        lstm_out, _ = self.lstm(inputs)
        # lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        out_space = self.hidden2out(lstm_out.view(len(inputs),-1))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores

def train(tensors, labels):
    print('Training')
    model = mylstm(17,17,9)
    model.to(torch.device('cuda'))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    tensors = tensors.to(torch.device('cuda'))
    labels = labels.to(torch.device('cuda'))

    print('Get initial model outputs')
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        out_scores = model(tensors)
        # out_scores = model(tensors[0].view(1,1,17))
        print(out_scores.shape)

    print('Enter training loop')
    loss = None
    for epoch in range(1000):
        print(loss)
        i = random.randint(0,len(tensors)-10001)
        x = tensors[i:i+10000]
        y = labels[i:i+10000]

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        yhat = model(x)
        # print('y',y.shape)
        # print('yt',y.view(-1).shape)
        # print('x',x.shape)
        # print('yh',yhat.shape)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(yhat, y)
        loss.backward()
        optimizer.step()

    print('Finish training')
    # See what the scores are after training
    with torch.no_grad():
        print('Evaluate model on dataset')
        yhat = torch.argmax(model(tensors), dim=1).int()
        print(yhat.shape)

        print('Get data from gpu')
        yhat = yhat.cpu().numpy()
        tensors = tensors.cpu().numpy()
        labels = labels.cpu().numpy()

        print('Begin generating labels')
        yhat_not_nolabel = yhat[yhat != 8]
        print(yhat_not_nolabel)
        print(yhat_not_nolabel.shape)



# def onehot(i,n):
#     return torch.tensor([[int(i == k) for k in range(n)]])

if  __name__ == '__main__':
    files = []
    for (dirpath, dirnames, filenames) in os.walk('precomputed/features'):
        files.extend(dirpath + '/' + f for f in filenames)
    print(files)

    NUMLABELS=9
    NOLABEL = torch.tensor([8])
    ENDSIGNAL = torch.tensor([[[-1]*17]], dtype=torch.float)
    print('endsignal.shape',ENDSIGNAL.shape)
    pick = {'lcRel':torch.tensor([0]),
            'evtEnd':torch.tensor([1]),
            'rightTO':torch.tensor([2]),
            'cutout':torch.tensor([3])}
    frame2gtLabel = {
        int(34.1177184317544*30+.5) : pick['lcRel'],
        int(40.068483274502285*30+.5) : pick['evtEnd'],
        int(46.336862502853776*30+.5) : pick['rightTO'],
        int(57.12734249037953*30+.5) : pick['lcRel'],
        int(62.24097239727076*30+.5) : pick['evtEnd'],
        int(62.258823252764785*30+.5) : pick['rightTO'],
        int(74.41899688556654*30+.5) : pick['cutout'],
        int(78.29332897516178*30+.5) : pick['evtEnd'],
        int(78.380841109407*30+.5) : pick['rightTO'],
    }

    # (objectid,x1,y1,x2,y2,*pvalues,lanescores)
    tensors = []
    labels = []
    with open(files[0], 'rb') as file:
        (allboxes, allboxscores, alllines, alllinescores, allvehicles, allprobs) = pickle.load(file)
        for i,(vehicles,probs,linescores) in enumerate(zip(allvehicles, allprobs, alllinescores)):
            crrectedls = [float(ls.cpu().numpy()) for ls in linescores]

            for vehicle,probl,probr in zip(vehicles,probs[:len(vehicles)],probs[len(vehicles):]):
                tensors.append(torch.tensor([[[*vehicle,*probl,*probr,*crrectedls]]],dtype=torch.float))
                labels.append(NOLABEL)

            # entice the network to produce a label
            tensors.append(ENDSIGNAL)
            if i in frame2gtLabel:
                labels.append(frame2gtLabel[i])
            else:
                labels.append(NOLABEL)

        tensors = torch.cat(tensors)
        labels = torch.cat(labels)
        train(tensors,labels)
