import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from puyo.Duel import *
from puyo.PuyoGame import PuyoGame as Game
from multiprocessing import Process, Queue, Manager
from puyo.PuyoGame import PuyoGame as Game
from puyo.pytorch.NNet import NNetWrapper as nn

import numpy as np
from tqdm import tqdm
from utils import *

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 1000,  # 1000
    # Number of complete self-play games to simulate during a new iteration.
    'numEps': 100,  # 100
    'tempThreshold': 15,        #
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'updateThreshold': 0.6,
    # Number of game examples to train the neural networks.
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    # Number of games to play during arena play to determine if new net will be accepted.
    'arenaCompare': 10,
    'cpuct': 3,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

proreturn = {}
threads = 4
nnet = nn(Game())

def executeEpisode(pn, args, returndict):
    global proreturn, nnet, mcts
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
    """
    game = Game()
    nnet = nnet
    mcts = MCTS(Game(), nnet, args)
    if args.load_model:
        #log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(
            args.load_folder_file[0], args.load_folder_file[1])
    else:
        pass
        #log.warning('Not loading a checkpoint!')
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        temp = int(episodeStep < args.tempThreshold)

        pi = mcts.getActionProb(
            game.getCanonicalFormBoard(board, curPlayer), temp=temp)
        #pi = self.mcts.getActionProb(Duel(duel=board), temp=temp)
        trainExamples.append(
            [board.GrayScaleArray(board.getGameInfo(0 if curPlayer == 1 else 1)), curPlayer, pi, None])

        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(
            board, curPlayer, action)

        r = game.getGameEnded(board, curPlayer)

        if r != 0:
            returndict[pn] = [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
            return

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self):
        global mcts, args
        self.args = args
        self.pnet = nnet.__class__(Game())
        self.game = Game()
        if(args.load_model == True):
            log.info("Loading 'trainExamples' from file...")
            self.loadTrainExamples()
        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()


    def learn(self):
        global proreturn, nnet, threads, mcts
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque(
                    [], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(0,self.args.numEps, threads), desc="Self Play"):
                    # reset search tree
                    pro = []
                    manager = Manager()
                    returndict = manager.dict()
                    for a in range(threads):
                        args = self.args
                        pro.append(Process(target=executeEpisode, args=(i, args,returndict) ))
                    for a in pro:
                        a.start()
                    for a in pro:
                        a.join()

                    for a in returndict.values():
                        iterationTrainExamples += a

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            nnet.train(trainExamples)
            nmcts = MCTS(self.game, nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                     (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
