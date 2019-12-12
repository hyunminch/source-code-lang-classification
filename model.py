from torch import nn
from torch.nn import functional as F
from torchtext.data import Field
from torchtext.vocab import Vectors
import numpy as np


class TextCNN(nn.Module):
    '''
    TextCNN is the CNN module of this neural network.
    It receives text tensors and sends them through 3 layers of 
    convolution filters that are coupled with max pooling.

    In the end, the output is sent to a fully-connected layer to
    5 outputs that denote the probability of the input to be
    the ith output.
    '''
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

        '''
        The commented out definitions of convs and
        fcs denote the layers that have been used for the
        more deeply nested version of this CNN.
        However, these are commented out as they do not
        contribute much to the accuracy of this neural network.
        '''

        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=7, stride=1),
        #     nn.ReLU()
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=7, stride=1),
        #     nn.ReLU()
        # )

        self.fc1 = nn.Linear(8960, 5)
        
        # self.fc1 = nn.Sequential(
        #     nn.Linear(5888, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1)
        # )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1)
        # )

        # self.fc3 = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        
        return x