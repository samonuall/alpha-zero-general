import logging

import coloredlogs

from Coach import Coach
from poker.PokerGame import PokerGame as Game
from poker.pytorch.NNet import NNetWrapper as nn
import torch
import wandb
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

args = dotdict({
        'numIters': 5,
        'numEps': 50,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 50,
        'arenaCompare': 15,
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/poker-bot','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        "sendToHub": False,
        "num_cpu": 4,
        # Fixed hyperparams for NNet
        'lr': .0005,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'block_width': 256,
        'n_blocks': 1,
        "run_name": "main_run"
    })


def main():
    wandb.init(
        project="alpha-poker",
        config=args,
        name=args.run_name,
        reinit=True,
    )

    print('Loading %s...', Game.__name__)
    g = Game()

    print('Loading %s...', nn.__name__)
    nnet = nn(g, args)

    if args.load_model:
        print('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        print('Not loading a checkpoint!')

    print('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        print("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    print('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
