from Duel import Duel
from ISMCTS import MCTS
from nnet import NNetWrapper
from Puyo_NNET import PuyoNNet
import torch.optim as optim
import torch
from utils import *
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from random import *

sys.path.append('../../')


model = NNetWrapper(PuyoNNet())
puyo_duel = Duel(seed=randint(0, 65535))

for i in range(1):
    #### augmentation for making duel with same AI ####
    puyo_duel = Duel(seed=randint(0, 65535))

    puyo_duel.input(randint(0, 21), randint(0, 21))
    puyo_duel.run()
    # puyo_duel.print()

    while (True):
        # make copy since duel class can't be initialize after MCTS loop
        puyo_duel_copy = Duel(duel=puyo_duel)

        # find out action probability using MCTS based on UCT+ DQN
        prob = MCTS(puyo_duel_copy, model).getActionProb(0.9)
        print('MCTS!!!!!', prob)

        action = np.random.choice(22, 1, p=prob)
        print('chosen action:', action)

        status = puyo_duel.status()

        if status == 0 or status == 2:
            puyo_duel.input(action[0], -1)
        if status == 1:
            puyo_duel.input(-1, action[0])
        if status == 3 or status == 4 or status == 5 or status == -1:
            break
        puyo_duel.run()
        # puyo_duel.print()
    # training
    # model.train(examples)
