import sys
import os
import pickle
import numpy as np
from tqdm import tqdm

# Ensure the base directory is in the Python path
base_dir = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from poker.PokerGame import PokerGame
from poker.PokerPlayers import NaivePlayer
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

        while True:
            # Get the canonical representation of the board for the current player
            # The state stored should be the one the NN will see
            canonicalBoard = game.getCanonicalForm(board, curPlayer)

            # Determine which player's turn it is
            active_player = player1 if curPlayer == 1 else player2

            # Get the action from the NaivePlayer
            # NaivePlayer.play expects the raw board state, not canonical
            # Note: NaivePlayer might need adjustments if it expects specific player UUIDs
            #       or if the board state passed needs modification.
            #       For simplicity, we pass the raw board. If NaivePlayer relies
            #       on player index 0 always being 'itself', this might need adjustment
            #       depending on how PokerGame handles player turns and board states.
            #       Let's assume PokerGame handles the board state correctly for the player.
            action = active_player.play(board) # Pass the raw board

            # Create the policy vector (one-hot encoding of the chosen action)
            pi = np.zeros(action_size)
            pi[action] = 1.0

            # Store the canonical state, current player, and policy vector
            # We store the canonical state because that's what the NN will train on
            episode_examples.append([canonicalBoard, curPlayer, pi])

            # Get the next state
            board, curPlayer = game.getNextState(board, curPlayer, action)

            # Check if the game ended
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

if __name__ == "__main__":
    # Example usage: Generate data for 1000 games and save it
    # You might want to save it in a specific checkpoint directory
    checkpoint_dir = os.path.join(base_dir, 'alpha-zero-general', 'pretrained_data')
    data_file = os.path.join(checkpoint_dir, 'naive_pretrain_data_1k.pkl')

    generate_naive_data(num_games=1000, output_file=data_file)

    # You can load it later like this:
    # with open(data_file, "rb") as f:
    #     loaded_data = pickle.load(f)
    # print(f"Loaded {len(loaded_data)} examples.")
    # if loaded_data:
    #     print("First example state shape:", loaded_data[0][0].shape)
    #     print("First example policy:", loaded_data[0][1])
    #     print("First example value:", loaded_data[0][2])
