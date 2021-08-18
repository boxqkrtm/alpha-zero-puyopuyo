import logging
import math

import numpy as np
import Duel
import random

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, D, nnet):
        ''' 매 순간마다 duel class를 따로 저장하는 이유는 ISMCTS를 함에 있어서 오쟈마에 대한 랜덤성을 돌려야 하는데
        이를 기존 package 상에서는 determinization이 이루어지지 않기 때문에 대안책으로 한번 도달한 곳에서는 기존에 저장해 놓았던
        Duel을 불러와서 오쟈마가 다르게 떨어지지 않게 하기 위해서이다

        이를 위해서 새로 도달한 곳마다는 하드카피를 떠서 dictionary 형태로 넣어 주어야 한다.'''
        self.D = D
        self.Dsa = {}  # game field in that state Dsa
        self.nnet = nnet  # neural net
        self.gi = []

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s  1p이기면 1, 1p가 지면 -1
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        puyo duel. For robustness, we simulated MCTS robustrandom time with different random seeds
        for robustness.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        for i in range(50):
            self.search()

        flag = self.D.run()
        if flag == 0 or flag == 2:  # when 1p input needed at this turn
            self.gi = self.D.getGameInfo(0)
            turn = 1
        if flag == 1:
            # we flip the data as 2p's perspective at 2p's turn to find out maximizing opponent's win rate
            self.gi = self.D.getGameInfo(1)
            turn = -1
        else:
            # game end
            return

        s = [GetFieldInfo(self.gi), GetOuterFieldInfo(self.gi)]
        s = ''.join(s)

        # Search count
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in self.D.GetValidMoves()]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self):
        '''
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        중간중간 return하는 value는 1p기준에서의 value(expect reward)를 return함.

        NOTE: puyonet에서 prediction은 현재 하는 player의 기준에서 value와 policy가 정해지기 때문에
             누구 턴인지를 구분하는 turn 변수를 둬서 setting함.
        '''

        '''누구 턴인지를 run을 돌려서 확인한다.'''
        flag = self.D.run()
        #print('flag', flag)
        if flag == 0 or flag == 2:  # when 1p input needed at this turn
            self.gi = self.D.getGameInfo(0)
            turn = 1
        if flag == 1:
            # we flip the data as 2p's perspective at 2p's turn to find out maximizing opponent's win rate
            self.gi = self.D.getGameInfo(1)
            turn = -1
        else:
            # game end
            return
        field = np.array(self.D.GetFieldInfo(self.gi)).reshape(-1)
        outfield = np.hstack(np.array(self.D.GetOuterFieldInfo(self.gi)))
        total_state = np.concatenate([field, outfield])
        #print('total state',total_state)

        s = total_state.tostring()

        if s not in self.Es:
            self.Es[s] = self.D.GetGameEnded()

        if self.Es[s] != 0:
            # 이미 도달한 적이 있는 terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node  --> make prediction
            # prediction : 현재 턴에서 해야하는 플레이어의 기준으로 정한 policy 및 기대 reward
            self.Ps[s], v = self.nnet.predict(self.D)
            valid_moves = self.D.GetValidMoves_masking()
            print('valid:', valid_moves)
            for index, element in enumerate(self.Ps[s]):
                (self.Ps[s])[index] = (self.Ps[s])[index] * \
                    (valid_moves[index])  # masking invalid moves

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get
                # overfitting or something else. If you have got dozens or hundreds of these messages you should pay
                # attention to your NNet and/or training process.

                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valid_moves
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = self.D.GetValidMoves()
            self.Ns[s] = 0
            return turn * v

        valids = self.Vs[s]

        cur_best = -float('inf')
        best_act = -1

        '''UCT 트리 만드는 과정'''
        # pick the action with the highest upper confidence bound
        for a in range(22):
            print(a)
            if valids[a]:
                if (s, a) in self.Qsa:

                    '''Q-value를 산출한적이 있으면 그걸 갖다 쓴다'''
                    u = self.Qsa[(s, a)] + 0.2 * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
                else:
                    '''없으면 그냥 0 으로 두고 search를 하면서 업데이트를 한다.'''
                    u = 0.2 * self.Ps[s][a] * \
                        math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                '''update하면서 UCB가 달라지게 되는 경우에는 이를 갱신한다.'''

                if turn * u > turn * cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        '''시뮬레이션 돌려본 수인지 아닌지 판단'''
        if (N[(s, a)] > 0):
            '''그 액션을 한 경우의 Duel field를 꺼낸다'''
            v = self.search(Dsa[(s, a)])
        else:
            # deep_copy를 뜬 다음에 뿌요를 하나 두고, self.search logic을 계속 작동시킨다.
            D_next = Duel(duel=self.D)

            if turn == 1:
                D_next.input(a, 0)
            if turn == -1:
                D_next.input(0, a)

            flag2 = D_next.run()

            if flag2 == 0 or flag2 == 2:
                turn_next = 1
            if flag2 == 1:
                turn_next = -1

            '''MCTS의 iteration을 하면서 값을 return 해주게 되는데 이게 현재 state의 value이다.'''

            v = self.search(D_next)
            D_next_copy = Duel(duel=D_next)
            Dsa[(s, a)] = D_next_copy

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] +
                                v * turn_next) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v * turn_next
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return turn_next * v
