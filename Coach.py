import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import utils
from pickle import Pickler

log = logging.getLogger(__name__)

# monkey patch prints with logging
def print_with_log(*args, **kwargs):
    """
    Print function that uses logging instead of print.
    """
    log.info(" ".join(map(str, args)), **kwargs)

import builtins
builtins.print = print_with_log


# Needed so that arena can do parallel computation
class MCTSPlayer:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(game, nnet, args)
    
    def __call__(self, x, temp=0):
        return np.argmax(self.mcts.getActionProb(x, temp=temp))


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
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
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        # take away 4 because of other processes happening
        NUM_PROCESSES = self.args.num_cpu
        log.info(f'Using {NUM_PROCESSES} processes')
        
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                # Save current model for workers to load
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='iter_model.pth.tar')

                # Calculate episodes per worker
                eps_per_worker = [self.args.numEps // NUM_PROCESSES + (1 if j < self.args.numEps % NUM_PROCESSES else 0) 
                             for j in range(NUM_PROCESSES)]
                
                # Create workers
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                checkpoint_path = os.path.join(self.args.checkpoint, 'iter_model.pth.tar')

                if NUM_PROCESSES > 1:
                    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:

                        # define call to worker function
                        f = lambda num_eps, id: executor.submit(self.worker_func, 
                                                num_eps, 
                                                self.game, 
                                                checkpoint_path,
                                                self.nnet.__class__,
                                                self.args,
                                                id)
                        
                        futures = [f(num_eps, id) for id, num_eps in enumerate(eps_per_worker)]
                        for future in as_completed(futures):
                            iterationTrainExamples += future.result()
                else:
                     # single-threaded fallback for debugging
                    for id, num_eps in enumerate(eps_per_worker):
                        results = self.worker_func(
                            num_eps,
                            self.game,
                            checkpoint_path,
                            self.nnet.__class__,
                            self.args,
                            id
                        )
                        iterationTrainExamples += results

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

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            if self.args.sendToHub:
                # upload data
                with open(self.args.checkpoint + "trainExamples.pkl", "wb+") as f:
                    Pickler(f).dump(trainExamples)
                
                utils.upload_file_to_hf(self.args.checkpoint + "trainExamples.pkl", "trainExamples.pkl")
                
                # Upload the model to Hugging Face Hub
                utils.upload_file_to_hf(self.args.checkpoint + "temp.pth.tar", "temp.pth.tar")

                # wait until new model is available
                if utils.get_new_model(self.args.checkpoint, "new_model.pth.tar"):
                    log.info("New model available, loading it...")
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                else:
                    raise Exception("New model not available, exiting...")
            else:
                # training new network, keeping a copy of the old one
                print("Training new network...")
                self.nnet.train(trainExamples)

            
            nmcts = MCTSPlayer(self.game, self.nnet, self.args)
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTSPlayer(self.game, self.pnet, self.args)
            
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(nmcts, pmcts, self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, num_workers=NUM_PROCESSES)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    @staticmethod
    def worker_func(num_eps, game, checkpoint_path, nnet_class, args, id):
        # set random seed so that each process will have different results
        np.random.seed(os.getpid())
        
        # Initialize network
        nnet = nnet_class(game, args)
        nnet.load_checkpoint(folder=os.path.dirname(checkpoint_path), 
                            filename=os.path.basename(checkpoint_path))

        # Collect examples from all episodes
        all_examples = []

        for _ in tqdm(range(num_eps), position=id):
            # reset tree
            mcts = MCTS(game, nnet, args)

            # Execute a single episode
            episode_examples = []
            board = game.getInitBoard()
            curPlayer = 1
            episodeStep = 0
            
            community_found = False
            while True:
                episodeStep += 1
                canonicalBoard = game.getCanonicalForm(board, curPlayer)
                temp = int(episodeStep < args.tempThreshold)
                
                pi = mcts.getActionProb(canonicalBoard, temp=temp)
                sym = game.getSymmetries(canonicalBoard, pi)
                
                for b, p in sym:
                    episode_examples.append([b, curPlayer, p, None])
                    
                action = np.random.choice(len(pi), p=pi)
                board, curPlayer = game.getNextState(board, curPlayer, action)
                
                r = game.getGameEnded(board, curPlayer)
                if r != 0:
                    # Process end-of-game results
                    # SAM: this is where I would need to change the reward part
                    processed = [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in episode_examples]
                    all_examples.extend(processed)
                    break
                    
        return all_examples
    
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
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