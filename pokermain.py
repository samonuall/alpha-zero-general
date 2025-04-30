import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

from Coach import Coach
from poker.PokerGame import PokerGame as Game
from poker.pytorch.NNet import NNetWrapper as nn
import torch
import wandb
from utils import *


# two more experiments tn:
# same but 80 dim
# 64 dim with larger numEps, batch size, and learning rate
# Maybe the 96 dim with same params, but 25 mctssims? Only if have time
# Tmrw ill start running with the most promising one in background
run_name = "lr_decay-board_texture-80-redo" # Change this to your desired run name
args = dotdict({
        'numIters': 10, # In all we should do at least 10_000 games
        'numEps': 2, # Turn this up to like 50 at least
        'tempThreshold': 15,
        'updateThreshold': 0.5,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 10,
        'arenaCompare': 30,
        'cpuct': 1,
        'checkpoint': f'./temp-{run_name}/',
        'load_model': False,
        'load_folder_file': ('/dev/models/poker-bot','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        "sendToHub": False,
        "num_cpu": 2,
        # Fixed hyperparams for NNet
        'lr': .00025, # Turn up lr for lower dimensions and batch sizes
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64, # probably increase batch size for final training
        'cuda': False,
        'block_width': 256,
        'n_blocks': 1,
        # Add wandb specific args
        "use_wandb": False, # Control wandb usage
        "wandb_project": "alpha-poker",
        "wandb_run_name": run_name # Use this for the run name
    })


def main():
    # Initialize wandb only if use_wandb is True
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, # Use project from args
            config=args,
            name=args.wandb_run_name, # Use run name from args
            reinit=True, # Keep reinit=True if needed, though likely only one init now
        )

    print('Loading %s...', Game.__name__)
    g = Game(seed=50)

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
