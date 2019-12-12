import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchtext.data import Field
from torchtext.vocab import Vectors

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt   

from dataset import CodeDataset
from model import TextCNN

alphabet_size = 128
num_classes = 5
num_epochs = 20

batch_size = 5
num_workers = 4


model = TextCNN(alphabet_size, num_classes)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(langs):
    dataset = CodeDataset("data/train", langs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_nr, data in enumerate(dataloader):
            inputs, target = data          

            # Assume cuda is on
            inputs, target = inputs.cuda(), target.cuda()

            inputs = Variable(inputs)
            target = Variable(target)
            output = model(inputs)

            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data

        epoch_loss = running_loss / len(dataloader)
        print('Epoch {} Loss: {:.4f}'.format(epoch + 1, epoch_loss))

def test(langs):
    dataset = CodeDataset("data/test", langs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

    predicteds = []
    targets = []

    for batch_nr, data in enumerate(dataloader):
        inputs, target = data          

        # Assume cuda is on
        inputs, target = inputs.cuda(), target.cuda()

        inputs = Variable(inputs)
        target = Variable(target)
        output = model(inputs)

        maxed = torch.max(output.cpu().data, 1)
        predicted = maxed[1] 

        predicteds.extend(predicted.numpy())
        targets.extend(target.cpu().numpy())

    y_preds = [langs[idx] for idx in predicteds]
    y_targets = [langs[idx] for idx in targets]

    score = accuracy_score(y_targets, y_preds)
    cm = confusion_matrix(y_targets, y_preds, langs)
    
    print("Score:", score)

    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=langs, yticklabels=langs)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=True)

if __name__ == '__main__':
    with open('langs.txt') as f:
        lines = [line.rstrip('\n') for line in f]
        langs = lines

    train(langs)
    model.eval()
    test(langs)
