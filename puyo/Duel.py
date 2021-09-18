# cython: language_level=3
# -*- coding: utf-8 -*-

from ctypes import *
import numpy as np
import random
import matplotlib.pyplot as plt

EPS = 1e-8

dueldll = 0
try:
  dueldll = cdll.LoadLibrary('./puyo/tuyotuyo')
except:
  dueldll = cdll.LoadLibrary('/usr/lib/tuyotuyo.so')
  

gcnt = 0

class GameInfo(object):

    def __init__(self, obj):
        global gcnt
        self.obj = obj
        gcnt += 1
        #print(gcnt)

    def __del__(self):
        global gcnt
        gcnt -= 1
        dueldll.GameInfoDel.argtypes = [c_void_p]
        dueldll.GameInfoDel.restype = c_void_p
        dueldll.GameInfoDel(self.obj)

dcnt = 0

class Duel(object):

    def __init__(self, seed=None, duel=None, obj=None):
        global dcnt
        #print(dcnt)
        self.p1 = -1
        self.p2 = -1
        self.isPlayer = 1
        dcnt += 1
        '''
        Duel 오브젝트를 입력시 복제, 시드입력시 새 Duel객체를 생성합니다
        '''
        if(seed != None):
            # create object
            dueldll.DuelNew.argtypes = [c_int]
            dueldll.DuelNew.restype = c_void_p
            if(seed == None):
                seed = random.randint(0, 65535)
            self.obj = dueldll.DuelNew(seed)
            self.input(0, 0)
            self.run()  # init
        elif (obj != None):
            # get object
            self.obj = obj.obj
        elif(duel != None):
            # copy object
            dueldll.DuelCopy.argtypes = [c_void_p]
            dueldll.DuelCopy.restype = c_void_p
            self.obj = dueldll.DuelCopy(duel.obj)
            self.p1 = duel.p1
            self.p2 = duel.p2
            self.isPlayer = duel.isPlayer
        else:
            # create object
            if(seed == None):
                seed = random.randint(0, 65535)
            dueldll.DuelNew.argtypes = [c_int]
            dueldll.DuelNew.restype = c_void_p
            self.obj = dueldll.DuelNew(seed)
            self.input(0, 0)
            self.run()  # init

    def reset(self, seed=None):
        if(seed == None):
            seed = random.randint(0, 65535)
        dueldll.DuelReset.argtypes = [c_void_p, c_int]
        dueldll.DuelReset.restype = c_void_p
        dueldll.DuelReset(self.obj, seed)

    def swap(self):
        # 1p 2p swap
        dueldll.DuelSwap.argtypes = [c_void_p]
        dueldll.DuelSwap.restype = c_void_p
        dueldll.DuelSwap(self.obj)
        tmp = self.p1
        self.p1 = self.p2
        self.p2 = tmp
        self.isPlayer = -self.isPlayer

    def print(self):
        dueldll.DuelPrint.argtypes = [c_void_p]
        dueldll.DuelPrint.restype = c_void_p
        dueldll.DuelPrint(self.obj)

    def run(self):
        '''
        --return--
        0 1p입력필요
        1 2p입력필요
        2 둘다 입력 필요
        3 1p승리
        4 2p승리
        5 무승부
        -1 에러
        '''
        dueldll.DuelRun.argtypes = [c_void_p]
        dueldll.DuelRun.restype = c_int
        return dueldll.DuelRun(self.obj)

    def status(self):
        '''
        --return--
        0 1p입력필요
        1 2p입력필요
        2 둘다 입력 필요
        3 1p승리
        4 2p승리
        5 무승부
        -1 에러
        '''
        dueldll.DuelStatus.argtypes = [c_void_p]
        dueldll.DuelStatus.restype = c_int
        return dueldll.DuelStatus(self.obj)

    def runWithOjamaSim(self, playerNum=0):
        '''
        playerNum:
            플레이어 번호 0은 1p, 1은 2p
        --return--
        6 받아올 방뿌시뮬레이션들 있음
        나머진 위의 run과 같음
        '''
        dueldll.DuelRunWithOjamaSim.argtypes = [c_void_p, c_int]
        dueldll.DuelRunWithOjamaSim.restype = c_int
        return dueldll.DuelRunWithOjamaSim(self.obj, playerNum)

    def getOjamaSim(self):
        '''
        위의 runWithOjamaSim에서 6을 받았을 경우
        생성된 방뿌 시뮬레이션 데이터들(Duel 객체)을
        Duel class들이 담긴 리스트로 받아옴 (6~20 size)
        '''
        dueldll.DuelGetOjamaSim.argtypes = [c_void_p, c_int]
        dueldll.DuelGetOjamaSim.restype = c_void_p

        result = []
        for i in range(20):
            object = dueldll.DuelGetOjamaSim(self.obj, i)
            if (object == None):
                break
            result.append(Duel(obj=object))

        self.resetOjamaSim()
        return result

    def resetOjamaSim(self):
        dueldll.DuelResetOjamaSim.argtypes = [c_void_p]
        dueldll.DuelResetOjamaSim.restype = c_void_p
        dueldll.DuelResetOjamaSim(self.obj)

    def getGameInfo(self, playerNum):
        dueldll.DuelGameInfo.argtypes = [c_void_p]
        dueldll.DuelGameInfo.restype = c_void_p
        return GameInfo(dueldll.DuelGameInfo(self.obj, playerNum))

    def input(self, p1Input, p2Input):
        '''
        input range is
        0~21
        -1 force turn skip function
        -2 random drop without error
        '''
        dueldll.DuelInput.argtypes = [c_void_p, c_int, c_int]
        dueldll.DuelInput.restype = c_void_p
        return dueldll.DuelInput(self.obj, p1Input, p2Input)

    def inputTest(self, playerNum, input):
        dueldll.DuelInputTest.argtypes = [c_void_p, c_int, c_int]
        dueldll.DuelInputTest.restype = c_bool
        return dueldll.DuelInputTest(self.obj, playerNum, input)

    def getMyField(self, gameInfo):
        gameInfo = gameInfo.obj
        '''
        1d array(13x6)
        index 0 is left-top
        '''
        dueldll.GameInfoGetMyField.argtypes = [c_void_p]
        dueldll.GameInfoGetMyField.restype = POINTER(c_int32)
        vals = dueldll.GameInfoGetMyField(gameInfo)
        valList = [vals[i] for i in range(13*6)]
        return valList

    def getOppField(self, gameInfo):
        gameInfo = gameInfo.obj
        '''
        1d array(13x6)
        index 0 is left-top
        '''
        dueldll.GameInfoGetOppField.argtypes = [c_void_p]
        dueldll.GameInfoGetOppField.restype = POINTER(c_int32)
        vals = dueldll.GameInfoGetOppField(gameInfo)
        valList = [vals[i] for i in range(13*6)]
        return valList

    def GameInfoGetMy14remove(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetMy14remove.argtypes = [c_void_p]
        dueldll.GameInfoGetMy14remove.restype = POINTER(c_bool)
        vals = dueldll.GameInfoGetMy14remove(gameInfo)
        valList = [vals[i] for i in range(6)]
        return valList

    def GameInfoGetOpp14remove(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetOpp14remove.argtypes = [c_void_p]
        dueldll.GameInfoGetOpp14remove.restype = POINTER(c_bool)
        vals = dueldll.GameInfoGetOpp14remove(gameInfo)
        valList = [vals[i] for i in range(6)]
        return valList

    def GameInfoGetMyAllClear(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetMyAllClear.argtypes = [c_void_p]
        dueldll.GameInfoGetMyAllClear.restype = c_bool
        return dueldll.GameInfoGetMyAllClear(gameInfo)

    def GameInfoGetOppAllClear(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetOppAllClear.argtypes = [c_void_p]
        dueldll.GameInfoGetOppAllClear.restype = c_bool
        return dueldll.GameInfoGetOppAllClear(gameInfo)

    def GameInfoGetMyNext(self, gameInfo):
        gameInfo = gameInfo.obj
        '''
        0 is air
        12345 is RGBYP
        6 is garbage
        '''
        dueldll.GameInfoGetMyNext.argtypes = [c_void_p]
        dueldll.GameInfoGetMyNext.restype = POINTER(c_int)
        vals = dueldll.GameInfoGetMyNext(gameInfo)
        valList = [vals[i] for i in range(4)]
        return valList

    def GameInfoGetOppNext(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetOppNext.argtypes = [c_void_p]
        dueldll.GameInfoGetOppNext.restype = POINTER(c_int)
        vals = dueldll.GameInfoGetOppNext(gameInfo)
        valList = [vals[i] for i in range(4)]
        return valList

    def GameInfoGetOppNext(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetOppNext.argtypes = [c_void_p]
        dueldll.GameInfoGetOppNext.restype = POINTER(c_int)
        vals = dueldll.GameInfoGetOppNext(gameInfo)
        valList = [vals[i] for i in range(4)]
        return valList

    def GameInfoGetMyOjama(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetMyOjama.argtypes = [c_void_p]
        dueldll.GameInfoGetMyOjama.restype = c_int
        return dueldll.GameInfoGetMyOjama(gameInfo)

    def GameInfoGetOppOjama(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetOppOjama.argtypes = [c_void_p]
        dueldll.GameInfoGetOppOjama.restype = c_int
        return dueldll.GameInfoGetOppOjama(gameInfo)

    def GameInfoGetMyEventFrame(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoGetMyEventFrame.argtypes = [c_void_p]
        dueldll.GameInfoGetMyEventFrame.restype = c_int
        return dueldll.GameInfoGetMyEventFrame(gameInfo)

    def GameInfoGetOppEventFrame(self, gameInfo):
        dueldll.GameInfoGetOppEventFrame.argtypes = [c_void_p]
        dueldll.GameInfoGetOppEventFrame.restype = c_int
        return dueldll.GameInfoGetOppEventFrame(gameInfo)

    def GetFieldInfo(self, gameInfo):
        gameInfo = gameInfo.obj
        return [self.getMyField(gameInfo), self.getOppField(gameInfo)]

    def GetOuterFieldInfo(self, gameInfo):
        gameInfo = gameInfo.obj
        return [self.GameInfoGetMyNext(gameInfo), self.GameInfoGetOppNext(gameInfo), self.GameInfoGetMyOjama(gameInfo), self.GameInfoGetOppOjama(gameInfo), self.GameInfoGetMyAllClear(gameInfo), self.GameInfoGetOppAllClear(gameInfo), self.GameInfoGetMyEventFrame(gameInfo), self.GameInfoGetOppEventFrame(gameInfo)]

    def GetValidMovesPlayer(self, player):
        validmoves = []
        for i in range(22):
            if self.inputTest(player, i) == True:
                validmoves += [i]
        return np.array(validmoves)

    def GetValidMovesPlayerMask(self, player):
        validmoves = []
        for i in range(22):
            if self.inputTest(player, i) == True:
                validmoves += [1]
            else:
                validmoves += [0]

        return np.array(validmoves)

    def GetGameEnded(self):
        copycat = Duel(duel=self)
        state = copycat.run()
        if state == 3:
            return 1
        if state == 4:
            return -1
        else:
            return 0

    def __del__(self):
        global dcnt
        dcnt -= 1
        dueldll.DuelDel.argtypes = [c_void_p]
        dueldll.DuelDel.restype = c_void_p
        dueldll.DuelDel(self.obj)
        del self.obj

    def GrayScaleArray(self, gameInfo):
        gameInfo = gameInfo.obj
        dueldll.GameInfoToGrayScale.argtypes = [c_void_p]
        dueldll.GameInfoToGrayScale.restype = POINTER(c_float)
        vals = dueldll.GameInfoToGrayScale(gameInfo)
        result = [vals[i] for i in range(14*14)]
        result = np.reshape(result, (14, 14, 1))
        self.GrayScaleDel(vals)
        #plt.imshow(result)
        #plt.show()
        return result

    def GrayScaleDel(self, obj):
        dueldll.GrayScaleDel.argtypes = [POINTER(c_float)]
        dueldll.GrayScaleDel.restype = c_void_p
        dueldll.GrayScaleDel(obj)
