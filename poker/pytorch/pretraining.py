import sys
import os
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.environ["BASE_DIR"] + "/alpha-zero-general/")
print(sys.path)

from poker.PokerGame import PokerGame
from poker.PokerPlayers import NaivePlayer
import NNet as nn
from utils import dotdict
# Assuming PokerState is implicitly handled by PokerGame methods or not directly needed here
# from poker.PokerLogic import PokerState # If direct state manipulation is needed

def generate_naive_data(num_games=100, output_file="naive_pretrain_data.pkl"):
    """
    Generates training data by simulating games between two NaivePlayers.

    Args:
        num_games (int): The number of games to simulate.
        output_file (str): The file path to save the generated data.

    Returns:
        list: A list of training examples, where each example is
              (canonical_state_vector, policy_vector, value).
    """
    game = PokerGame()
    player1 = NaivePlayer(game)
    player2 = NaivePlayer(game)
    action_size = game.getActionSize()

    all_training_data = []

    print(f"Simulating {num_games} games between two NaivePlayers...")
    for _ in tqdm(range(num_games)):
        board = game.getInitBoard()
        curPlayer = 1
        episode_examples = [] # Stores (canonical_state, current_player, policy_vector) for the current game
        players = [player2, None, player1]
        
        while True:
            action = players[curPlayer + 1].play(game.getCanonicalForm(board, curPlayer))
            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                raise ValueError(f"Invalid action {action} for player {curPlayer}!")
            
            episode_examples.append([
                game.getCanonicalForm(board, curPlayer), # Store the canonical state
                curPlayer, # Store the current player
                [int(action == i) for i in range(4)] # Store the policy vector (valid actions)
            ])
            

            board, curPlayer = game.getNextState(board, curPlayer, action)
            
            reward = game.getGameEnded(board, curPlayer) # Pass the *next* player
            if reward != 0:
                # Game has ended, assign values to all stored examples
                processed_examples = []
                for state, example_player, policy in episode_examples:
                    # The value is the final reward from the perspective of the player who made the move
                    # reward is from the perspective of the *next* player (curPlayer)
                    value = reward * ((-1) ** (example_player != curPlayer))
                    # Convert state (PokerState object) to vector for NN input
                    state_vector = state.to_vector()
                    processed_examples.append((state_vector, policy, value))

                all_training_data.extend(processed_examples)
                break # End the current game simulation

    print(f"Generated {len(all_training_data)} training examples.")

    # Save the data
    if output_file:
        print(f"Saving training data to {output_file}...")
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, "wb") as f:
            pickle.dump(all_training_data, f)
        print("Data saved.")

    return all_training_data


NUM_GAMES = 1000
if __name__ == "__main__":
    # Example usage: Generate data for 1000 games and save it
    # You might want to save it in a specific checkpoint directory
    checkpoint_dir = os.path.join(os.environ["BASE_DIR"], 'alpha-zero-general', 'pretrained_data')
    data_file = os.path.join(checkpoint_dir, 'naive_pretrain_data_1k.pkl')

    # generate_naive_data(num_games=NUM_GAMES, output_file=data_file)

    with open(data_file, "rb") as f:
        loaded_data = pickle.load(f)
    
    print(f"Loaded {len(loaded_data)} examples.")
    

    # Begin training
    game = PokerGame()
    args = dotdict({
        # Fixed hyperparams for NNet
        'lr': .00025, # Turn up lr for lower dimensions and batch sizes
        'epochs': 10,
        'batch_size': 512, # probably increase batch size for final training
        'cuda': False,
        "use_wandb": False, # Control wandb usage
    })
    model = nn.NNetWrapper(game, args)

    class Fake:
        def __init__(self, data):
            self.data = data

        def to_vector(self):
            return self.data

    model.train([(Fake(state), pi, v) for state, pi, v in loaded_data])
    # Save the model
    model.save_checkpoint(folder=checkpoint_dir, filename='naive_pretrained_model_100.pth.tar')
