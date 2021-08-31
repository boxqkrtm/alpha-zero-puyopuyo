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
        # 1p 2p 둘다 판단
        if(state == 2):
            # 놓기판단
            if(player == 1):
                #print("set 1p ", action, b.p2)
                b.p1 = action
            else:
                #print("set 2p ", action, b.p1)
                b.p2 = action

            # 둘다 판단을 완료해야 진행 아니면 턴만 넘기고 진행안함
            if(b.p1 != -1 and b.p2 != -1):
                b.input(b.p1, b.p2)
                b.run()
                b.p1, b.p2 = -1, -1
            else:
                return (b, -player)

        elif(state == 0 and player == 1):
            # 1p 놓기
            b.input(action, 0)
            b.run()
        elif(state == 1 and player == -1):
            # 2p 놓기
            b.input(0, action)
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
            return (b, 1)  # game over

    def getValidMoves(self, board, player):
        if(player == 1):
            return board.GetValidMovesPlayerMask(0)
        else:
            return board.GetValidMovesPlayerMask(1)

    def getGameEnded(self, boarda, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        isflip = False
        board = Duel(duel=boarda)
        if(player == -1):
            board = self.getCanonicalFormBoard(board, -1)
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


    def getCanonicalFormBoard(self, board, player):
        b = Duel(duel=board)
        # return state if player==1, else return -state if player==-1
        if(1 == player):
            return b
        else:
            b.swap()
            return b

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
