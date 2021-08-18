from __future__ import print_function
import numpy as np
from Game import Game
from .Duel import *
import sys
sys.path.append('..')


class PuyoGame(Game):

    def __init__(self):
        # 플레이어의 Action값 저장
        pass

    def getInitBoard(self):
        # return board object
        return Duel()

    def getBoardSize(self):
        # (a,b) tuple
        return (14, 14)

    def getActionSize(self):
        # return number of actions
        return 22

    def getNextState(self, board, player, action):
        b = Duel(duel=board)

        #b.input(action, -1)
        state = b.status()
        if(state == 2):
            # 둘다 놔야댐 1p
            if(b.p1 == -1):
                b.p1 = action
            else:
                b.p2 = action
                b.input(b.p1, b.p2)
                b.run()
                b.p1, b.p2 = -1, -1
                return (b, 1)
            return (b, -1)  # 자신의 입력 설정 후 2p로 넘겨줌
        elif(state == 0):
            # 1p 놓기
            b.input(action, b.p2)
            b.run()
        elif(state == 0):
            # 2p 놓기
            b.input(b.p1, action)
            b.run()

        # state 판단하여 board 돌려줌
        state = b.status()
        # 플레이어 턴 판단
        if(state == 0):
            return (b, 1)
        elif(state == 1):
            return (b, -1)
        elif(state == 2):  # 둘다 필요할땐 1p로 설정
            return (b, 1)
        else:
            return (b, 0)  # game over

    def getValidMoves(self, board, player):
        if(player == 1):
            return board.GetValidMovesPlayerMask(0)
        else:
            return board.GetValidMovesPlayerMask(1)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        state = board.status()
        if(state == 3):
            return 1
        elif(state == 4):
            return -1
        elif(state == 5):
            return -2  # draw
        else:
            # board.print()
            return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if(player == 1):
            return board.GrayScaleArray()
        else:
            return board.GrayScaleArray(board.getGameInfo(1))

    def getCanonicalFormBoard(self, board, player):
        # return state if player==1, else return -state if player==-1
        if(player == 1):
            return board
        else:
            board.swap()
            return board

    def getScore(self, board, player):
        state = board.status()
        if(state == 3):
            return 1*player
        elif(state == 4):
            return -1*player
        elif(state == 5):
            return 0  # draw

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board.GrayScaleArray())

    def display(self):
        self.d.print()
