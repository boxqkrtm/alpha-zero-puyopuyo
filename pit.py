import pyximport
pyximport.install()
from utils import *
from puyo.pytorch.NNet import NNetWrapper as NNet
from puyo.PuyoPlayers import *
from puyo.PuyoGame import PuyoGame as Game
from MCTS import MCTS
import Arena
import numpy as np
import sys
import os


os.system("chcp 65001")

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = False

g = Game()

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
# n1.load_checkpoint('./temp/', 'best.pth.tar')
n1.load_checkpoint('./temp/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct': 0.0})
mcts1 = MCTS(g, n1, args1)
def n1p(x): return np.argmax(mcts1.getActionProb(x, temp=0))


if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/', 'best.pth.tar')
    args2 = dotdict({'numMCTSSims': 25, 'cpuct': 0.10})
    mcts2 = MCTS(g, n2, args2)
    def n2p(x): return np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.
    #player2 = rp
while True:

    arena = Arena.Arena(n1p, player2, g, display=Game.display)

    print(arena.playGames(2, verbose=True))
