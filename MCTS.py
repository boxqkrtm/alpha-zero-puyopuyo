# cython: language_level=3
# cython: linetrace=True
# cython: profile=True
# cython: binding=True

import logging
import math
from puyo.Duel import *

import numpy as np
import time
import gc

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, board, temp=1):

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        d = Duel(duel=board)
        #self.search(d, depth=0)
        for i in range(self.args.numMCTSSims):
            d = Duel(duel=board)
            # d.print()
            # d = Duel()
            d.randUnknownData()
            self.search(d, depth=0)
        gi = board.getGameInfo(0)
        # s = str(board.GetFieldInfo(gi))+str(board.GetOuterFieldInfo(gi))
        depth = 0
        a = ""
        nplayer = board.isPlayer
        # s = str(depth)+"-"+str(a)+"-"+str(nplayer)
        s = str(board.GetFieldInfo(gi)) + \
            str(board.GetOuterFieldInfo(gi))+"-"+str(board.isPlayer)
        # self.game.stringRepresentation(board)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if(counts_sum == 0):
            d.print()
            print("S", s)
            print("a", a)
            print(self.Nsa[(s, a)])
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board, depth=0, first=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        gi = board.getGameInfo(0)
        # s = str(depth)+"-"+str(a)+"-"+str(board.isPlayer)
        s = str(board.GetFieldInfo(gi)) + \
            str(board.GetOuterFieldInfo(gi))+"-"+str(board.isPlayer)
        # self.game.stringRepresentation(board)
        # print(s)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, -1)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(
                board.GrayScaleArray(board.getGameInfo(0)))
            # v = self.game.getFieldOjama(board, -1)
            valids = self.game.getValidMoves(board, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                # board.print()
                print(valids)
                print(self.Ps[s])
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            if(depth < 2 and first == True):
                pass
            else:
                return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * \
                        self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
                if(first and depth == 0):
                    next_board, next_player = self.game.getNextState(
                        board, 1, a)
                    cb = self.game.getCanonicalFormBoard(
                        next_board, next_player)

                    v = self.search(cb, depth+1)

                    if (s, a) in self.Qsa:
                        self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                            self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                        self.Nsa[(s, a)] += 1

                    else:
                        self.Qsa[(s, a)] = v
                        self.Nsa[(s, a)] = 1

                    self.Ns[s] += 1

        a = best_act

        next_board, next_player = self.game.getNextStateRaw(board, 1, a)
        cb = self.game.getCanonicalFormBoardRaw(next_board, next_player)

        v = self.search(cb, depth+1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
