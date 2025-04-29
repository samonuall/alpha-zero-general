import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

import os, sys
sys.path.append(os.getcwd() + "/alpha-poker")

import itertools
import wandb
import numpy as np
import shutil # Added import

from Coach import Coach
from poker.PokerGame import PokerGame as Game
from poker.pytorch.NNet import NNetWrapper as nn
from poker.PokerPlayers import NaivePlayer
from Arena import Arena
from utils import dotdict

import torch
import time # Added import

# Define sweep values
cpuct_values = [0.5, 0.75, 1.0]
lr_values = [0.001, 0.0005, 0.0001]
numMCTSSims_values = [75, 100, 125]

# Baseline agent for Arena evaluation
def get_baseline_player(game):
    return NaivePlayer(game).play

def run_experiment(config, run_name=None):
    # Set up wandb
    wandb.init(
        project="poker-hyperparam-search",
        config=config,
        name=run_name,
        reinit=True,
    )

    # Prepare args for Coach and NNet
    args = dotdict({
        'numIters': 5,
        'numEps': 50,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': config['numMCTSSims'],
        'arenaCompare': 15,
        'cpuct': config['cpuct'],
        'checkpoint': f'./temp-{run_name}/',
        'load_model': False,
        'load_folder_file': ('/dev/models/poker-bot','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        "sendToHub": False,
        "num_cpu": 4,
        # Fixed hyperparams for NNet
        'lr': config['lr'],
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'block_width': 256,
        'n_blocks': 3,
        "run_name": run_name
    })

    g = Game()
    nnet = nn(g, a=args)
    c = Coach(g, nnet, args)

    # Run training
    start_time = time.time()
    c.learn()
    learn_time = time.time() - start_time

    # Evaluate in Arena
    g = Game(seed=42)  # Create seeded game so that each experiment will evaluate with same hands
    baseline_player = get_baseline_player(g)
    from MCTS import MCTS
    args_mcts = dotdict({'numMCTSSims': config["numMCTSSims"], 'cpuct': 1.0}) # Note: Arena MCTS sims are fixed here, training sims vary
    mcts = MCTS(g, nnet, args_mcts)
    nnet_player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

    arena = Arena(nnet_player, baseline_player, g)
    oneWon, twoWon, draws = arena.playGames(20, verbose=False, num_cpu=args.num_cpu)
    win_rate = 0
    if (oneWon + twoWon + draws) > 0:
        win_rate = oneWon / (oneWon + twoWon + draws)

    wandb.log({
        "arena_win_rate": win_rate,
        "learn_time_seconds": learn_time,
    })
    wandb.finish()

    # Clean up checkpoint directory
    try:
        shutil.rmtree(args.checkpoint)
        print(f"Removed checkpoint directory: {args.checkpoint}")
    except OSError as e:
        print(f"Error removing directory {args.checkpoint}: {e.strerror}")

    return win_rate # Return win rate for comparison

def main():
    best_cpuct = None
    best_lr = None
    best_numMCTSSims = None
    best_win_rate = -1
    fixed_lr = 0.0005
    default_numMCTSSims = 75 # Use a default for the first two sweeps

    print("--- Starting CPUCT Sweep ---")
    for i, cpuct_val in enumerate(cpuct_values):
        config = {
            'cpuct': cpuct_val,
            'lr': fixed_lr,
            'numMCTSSims': default_numMCTSSims
        }
        run_name = f"sweep1_cpuct_{cpuct_val}_lr_{fixed_lr}_sims_{default_numMCTSSims}"
        print(f"Running: {run_name} with config {config}")
        current_win_rate = run_experiment(config, run_name=run_name)
        print(f"Finished: {run_name} - Win Rate: {current_win_rate:.4f}")

        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            best_cpuct = cpuct_val
            print(f"*** New best cpuct: {best_cpuct} (Win Rate: {best_win_rate:.4f}) ***")

    if best_cpuct is None:
        print("Error: No successful runs in CPUCT sweep.")
        return

    print(f"\n--- Best CPUCT found: {best_cpuct} ---")
    print("--- Starting LR Sweep ---")
    best_win_rate = -1 # Reset for next sweep

    for i, lr_val in enumerate(lr_values):
        config = {
            'cpuct': best_cpuct,
            'lr': lr_val,
            'numMCTSSims': default_numMCTSSims
        }
        run_name = f"sweep2_cpuct_{best_cpuct}_lr_{lr_val}_sims_{default_numMCTSSims}"
        print(f"Running: {run_name} with config {config}")
        current_win_rate = run_experiment(config, run_name=run_name)
        print(f"Finished: {run_name} - Win Rate: {current_win_rate:.4f}")

        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            best_lr = lr_val
            print(f"*** New best lr: {best_lr} (Win Rate: {best_win_rate:.4f}) ***")

    if best_lr is None:
        print("Error: No successful runs in LR sweep.")
        return

    print(f"\n--- Best LR found: {best_lr} (with cpuct={best_cpuct}) ---")
    print("--- Starting numMCTSSims Sweep ---")
    best_win_rate = -1 # Reset for final sweep

    for i, sims_val in enumerate(numMCTSSims_values):
        config = {
            'cpuct': best_cpuct,
            'lr': best_lr,
            'numMCTSSims': sims_val
        }
        run_name = f"sweep3_cpuct_{best_cpuct}_lr_{best_lr}_sims_{sims_val}"
        print(f"Running: {run_name} with config {config}")
        current_win_rate = run_experiment(config, run_name=run_name)
        print(f"Finished: {run_name} - Win Rate: {current_win_rate:.4f}")

        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            best_numMCTSSims = sims_val
            print(f"*** New best numMCTSSims: {best_numMCTSSims} (Win Rate: {best_win_rate:.4f}) ***")

    if best_numMCTSSims is None:
        print("Error: No successful runs in numMCTSSims sweep.")
        return

    print("\n--- Sequential Hyperparameter Search Complete ---")
    print(f"Best cpuct: {best_cpuct}")
    print(f"Best lr: {best_lr}")
    print(f"Best numMCTSSims: {best_numMCTSSims}")
    print(f"Best Win Rate Achieved: {best_win_rate:.4f}")


if __name__ == "__main__":
    main()