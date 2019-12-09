from torch import nn
from torch.nn import functional as F
from torchtext.data import Field
from torchtext.vocab import Vectors
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, alphabet_size):
        super(TextCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(alphabet_size, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )            

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )            
        
        self.fc1 = nn.Sequential(
            nn.Linear(8960, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc3 = nn.Linear(4096, 5)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x