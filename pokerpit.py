import Arena
from MCTS import MCTS
from poker.PokerGame import PokerGame # Import PokerGame
from poker.PokerPlayers import * # Import PokerPlayers
from poker.pytorch.NNet import NNetWrapper as NNet # Import Poker NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

args = dotdict({
        'numIters': 10,
        'numEps': 25,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 20,
        'arenaCompare': 15,
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/poker-bot','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        "sendToHub": False,
        "num_cpu": 10,
        # Fixed hyperparams for NNet
        'lr': .0005,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': False,
        'block_width': 256,
        'n_blocks': 1,
        "wandb_run_name": "test_run",
        "use_wandb": False, # Set to True to use wandb
        "dim": 80,
        "numRandomSims": 20
    })

human_vs_cpu = True # Set to True to play against the AI

# Instantiate PokerGame
g = PokerGame()

# Instantiate Poker players
rp = RandomPlayer(g).play
gp = GreedyPokerPlayer(g).play
naive = NaivePlayer(g).play
hp = HumanPokerPlayer(g).play


model = NNet(g, args)
model.load_checkpoint(os.environ["BASE_DIR"], "alpha-zero-general/pretrained_data/naive_pretrained_model.pth.tar")
print(args.numMCTSSims)
pi, v = model.predict(g.getCanonicalForm(g.getInitBoard(), 1))
print("PredICT:", pi, v)
neural_player = NNetPlayer(PokerGame(), model, args).play

# nnet players
n1 = NNet(g, args)
# Comment out checkpoint loading as we likely don't have a pretrained Poker model yet
# n1.load_checkpoint('./pretrained_models/poker/pytorch/','some_poker_model.pth.tar')
args1 = dotdict({'numMCTSSims': 10, 'cpuct':1.0}) # Adjust sims as needed for Poker
args = dotdict(args1)
print(args)
args["numMCTSSims"] = 10
print
mcts1 = MCTS(g, n1, args)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    # Example for pitting against another NNet or a different player
    # n2 = NNet(g)
    # n2.load_checkpoint('./pretrained_models/poker/pytorch/', 'some_other_poker_model.pth.tar')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    # player2 = n2p

    # Pit against Greedy player for now if not human
    player2 = neural_player

# Define a display function for Poker (Arena expects a function)
# Using stringRepresentation for now, might need adjustment based on Arena's needs
def display_poker_board(board):
    pass
    # print(g.stringRepresentation(board))

# Pass the Poker game and display function to Arena
arena = Arena.Arena(neural_player, player2, g, display=display_poker_board)

# Play games (e.g., 2 games)
print(arena.playGames(2, verbose=True))
