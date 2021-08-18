'''

neural network structure only

reused code from

https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py

'''




import sys
sys.path.append('..')
from utils import *
import Duel
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



class PuyoNNet(nn.Module):
    def __init__(self):
        # game params
        # you should put

        super(PuyoNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*(13-4)*(12-4), 64)
        self.fc_bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64+14, 64)  # insert outerfield informations
        self.fc_bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(64, 32)
        self.fc_bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 22) # 22 = action size of single puyo

        self.fc5 = nn.Linear(22, 1)

    def forward(self, f_i, of_i): # f_i: field info // of_i: outerfield info as tensor


        #f_i: batch_size(game length normally) x board_x x board_y
        f_i = f_i.view(-1, 1, 13, 12)                                    # batch_size x 1 x board_x x board_y
        f_i = F.relu(self.bn1(self.conv1(f_i)))                          # batch_size x num_channels x board_x x board_y
        f_i = F.relu(self.bn2(self.conv2(f_i)))                          # batch_size x num_channels x board_x x board_y
        f_i = F.relu(self.bn3(self.conv3(f_i)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        f_i = F.relu(self.bn4(self.conv4(f_i)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)

        f_i = f_i.view(-1, 32*(13-4)*(12-4))

        f_i = F.dropout(F.relu(self.fc_bn1(self.fc1(f_i))), p=0.2, training=self.training)  # batch_size x 64

        of_i = of_i.view(-1, 14)
        f_i = torch.cat((f_i, of_i), dim=1)


        f_i = F.dropout(F.relu(self.fc_bn2(self.fc2(f_i))), p=0.2, training=self.training)  # batch_size x
        f_i = F.dropout(F.relu(self.fc_bn3(self.fc3(f_i))), p=0, training=self.training)  # batch_size x


        pi = self.fc4(f_i)                                                                         # batch_size x action_size
        v = self.fc5(pi)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


