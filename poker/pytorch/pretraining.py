import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing # Added for parallel processing
import random

sys.path.append(os.environ["BASE_DIR"] + "/alpha-zero-general/")
print(sys.path)

from poker.PokerGame import PokerGame
from poker.PokerPlayers import NaivePlayer, AggressivePlayer
import NNet as nn
from utils import dotdict
from pypokerengine.engine.data_encoder import DataEncoder

# Worker function for multiprocessing
def _simulate_single_game_worker(max_round):
    """
    Simulates a single game between two NaivePlayers and returns training examples.
    """
    game = PokerGame(max_round=max_round)
    action_size = game.getActionSize()

    # Create fresh players for each game simulation
    player1_sim = NaivePlayer(game)
    player2_sim = NaivePlayer(game)
    players = [player2_sim, None, player1_sim]

    board = game.getInitBoard()
    curPlayer = 1
    episode_raw_examples = []  # Stores (canonical_state_object, current_player_who_made_move, policy_vector)

    curr_round = 0
    try:
        while True:
            canonical_board = game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer + 1].play(canonical_board)
            
            # It's good practice to ensure the action is valid, though NaivePlayer should be correct.
            # valids = game.getValidMoves(canonical_board, 1)
            # if valids[action] == 0:
            #     # This would indicate an issue, potentially raise an error or log
            #     print(f"Warning: NaivePlayer chose invalid action {action} in worker.")
            #     # Depending on game rules, this might need specific handling.
            #     # For now, we assume NaivePlayer is well-behaved.

            episode_raw_examples.append([
                canonical_board,  # Store the PokerState object
                curPlayer,
                [int(action == i) for i in range(action_size)],
                0  # Placeholder for reward, to be filled later
            ])

            prevPlayer = curPlayer
            board, curPlayer = game.getNextState(board, curPlayer, action)
            if board.emulator_state["round_count"] > curr_round:
                # Calculate the reward for the previous player based on stack changes
                new_board = game.getCanonicalForm(board, curPlayer)
                prev_board = episode_raw_examples[-1][0]
                reward = list(new_board.player_states.values())[0]["my_stack"] - list(prev_board.player_states.values())[0]["my_stack"]
                reward /= 200
                if reward > 1:
                    reward = 1
                elif reward < -1:
                    reward = -1
                # if prevPlayer == 1:
                #     print(f"Reward for NaivePlayer: {reward}")
                # else:
                #     print(f"Reward for AggressivePlayer: {reward}")
                # Go back and assign reward to all previous actions in this round
                for i in range(len(episode_raw_examples) - 1, -1, -1):
                    if episode_raw_examples[i][0].emulator_state["round_count"] == curr_round:
                        episode_raw_examples[i][3] = reward * ((-1) ** (episode_raw_examples[i][1] != prevPlayer))
                    else:
                        break
                # Update the current round
                curr_round = board.emulator_state["round_count"]


                
            
            final_reward = game.getGameEnded(board, curPlayer)  # Pass the *next* player

            if final_reward != 0:  # Game has ended
                # Assign the final reward to the last round's actions
                new_board = game.getCanonicalForm(board, curPlayer)
                prev_board = episode_raw_examples[-1][0]
                reward = list(new_board.player_states.values())[0]["my_stack"] - list(prev_board.player_states.values())[0]["my_stack"]
                reward /= 200
                if reward > 1:
                    reward = 1
                elif reward < -1:
                    reward = -1
                for i in range(len(episode_raw_examples) - 1, -1, -1):
                    if episode_raw_examples[i][0].emulator_state["round_count"] == curr_round:
                        episode_raw_examples[i][3] = reward * ((-1) ** (episode_raw_examples[i][1] != prevPlayer))
                    else:
                        break
                
                processed_examples_for_this_game = []
                for state_obj, example_player, policy, reward in episode_raw_examples:
                    state_vector = state_obj.to_vector()  # Convert PokerState object to vector
                    processed_examples_for_this_game.append((state_vector, policy, reward))
                
                return processed_examples_for_this_game

    except Exception as e:
        print(f"Error during game simulation: {e}")
        # Handle any cleanup or logging if necessary
        return []

def generate_naive_data(num_games=100, max_round=1, output_file="naive_pretrain_data.pkl", num_cpus=1):
    """
    Generates training data by simulating games between two NaivePlayers, potentially in parallel.

    Args:
        num_games (int): The number of games to simulate.
        max_round (int): The maximum number of rounds per game.
        output_file (str): The file path to save the generated data.
        num_cpus (int): The number of CPU cores to use for parallel simulation.

    Returns:
        list: A list of training examples, where each example is
              (canonical_state_vector, policy_vector, value).
    """
    all_training_data = []

    print(f"Simulating {num_games} games using {num_cpus} CPUs with max_round={max_round}...")

    tasks_args = [max_round] * num_games

    if num_cpus > 1 and num_games > 0:
        # Ensure num_cpus does not exceed num_games if num_games is small
        effective_cpus = min(num_cpus, num_games)
        print(f"Using {effective_cpus} effective CPUs for parallel processing.")
        with multiprocessing.Pool(processes=effective_cpus) as pool:
            results_per_game = list(tqdm(pool.imap_unordered(_simulate_single_game_worker, tasks_args), total=num_games, desc="Simulating games"))
    elif num_games > 0: # Single CPU execution or num_cpus = 1
        print("Using single CPU for simulation.")
        results_per_game = []
        for i in tqdm(range(num_games), desc="Simulating games"):
            results_per_game.append(_simulate_single_game_worker(tasks_args[i]))
    else: # No games to simulate
        results_per_game = []


    # Flatten the list of lists
    for game_data_list in results_per_game:
        if game_data_list:  # Ensure game_data_list is not None or empty
            all_training_data.extend(game_data_list)
    
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


def test_nn(model, test_data):
    """
    Tests the neural network's performance on the provided test data.

    Args:
        model (NNet.NNetWrapper): The neural network model to test.
        test_data (list): A list of test examples, where each example is
                          (canonical_state_vector, policy_vector, value).
    """
    if not test_data:
        print("No test data provided.")
        return None, None, None

    policy_correct_predictions = 0
    sum_value_mse = 0.0
    sum_value_mae = 0.0
    num_samples = len(test_data)

    print(f"\nTesting model on {num_samples} samples...")

    for state_vector, true_pi_list, true_v in tqdm(test_data, desc="Evaluating model"):
        # state_vector is already a numpy array
        # true_pi_list is a list, true_v is a scalar

        pred_pi_raw, pred_v_raw = model.predict(state_vector)
        
        # pred_pi_raw is a numpy array (action_size,)
        # pred_v_raw is a numpy array, e.g., array([0.53]). Get scalar via .item() or [0]
        # Assuming pred_v_raw is like np.array([value]), common for PyTorch NNs outputting (batch,1) for value
        pred_v = pred_v_raw.item() if pred_v_raw.size == 1 else pred_v_raw[0]


        true_pi_array = np.array(true_pi_list)

        # Policy accuracy: check if argmax matches
        if np.argmax(pred_pi_raw) == np.argmax(true_pi_array):
            policy_correct_predictions += 1
        
        # Value accuracy
        sum_value_mse += (pred_v - true_v)**2
        sum_value_mae += abs(pred_v - true_v)

    policy_accuracy = (policy_correct_predictions / num_samples) * 100 if num_samples > 0 else 0
    avg_value_mse = sum_value_mse / num_samples if num_samples > 0 else float('inf')
    avg_value_mae = sum_value_mae / num_samples if num_samples > 0 else float('inf')

    print("\n--- Test Results ---")
    print(f"Policy Accuracy (argmax match): {policy_accuracy:.2f}%")
    print(f"Average Value MSE: {avg_value_mse:.4f}")
    print(f"Average Value MAE: {avg_value_mae:.4f}")
    print("--------------------")

    return policy_accuracy, avg_value_mse, avg_value_mae


NUM_GAMES = 2_000
NUM_CPU = 10
REGENERATE_DATA = True 
MAX_ROUND = 50
run_name = "2_000data_naiveplayer_50rounds"
data_file = "pretrain_data_50round_500games_naiveplayer2k.pkl"
if __name__ == "__main__":
    # Example usage: Generate data for 1000 games and save it
    # You might want to save it in a specific checkpoint directory
    checkpoint_dir = os.path.join(os.environ["BASE_DIR"], 'alpha-zero-general', 'pretrained_data')
    data_file = os.path.join(checkpoint_dir, data_file)

    # Set to True to regenerate data, False to load existing data
    if REGENERATE_DATA or not os.path.exists(data_file):
        print(f"Generating data with {NUM_GAMES} games...")
        generate_naive_data(num_games=NUM_GAMES, 
                                max_round=MAX_ROUND, 
                                output_file=data_file, 
                                num_cpus=NUM_CPU)
    else:
        print(f"Skipping data generation, {data_file} already exists.")


    with open(data_file, "rb") as f:
        loaded_data = pickle.load(f)
        print(f"Loaded {len(loaded_data)} training examples from {data_file}")
    
    random.shuffle(loaded_data)  # Shuffle the data

    # Split data into training and test sets
    test_split_ratio = 0.2 # Use 20% of data for testing
    num_test_samples = int(len(loaded_data) * test_split_ratio)

    if num_test_samples == 0 and len(loaded_data) > 0: # Ensure at least one test sample if data exists
        num_test_samples = 1 if len(loaded_data) > 1 else 0 # Or handle small datasets differently
    
    if len(loaded_data) == 0:
        print("No data loaded, cannot train or test.")
        sys.exit()

    if num_test_samples == 0 and len(loaded_data) > 0 : # if only one sample total
        print("Warning: Dataset too small for a meaningful train/test split. Using all data for training and testing.")
        train_examples = loaded_data
        test_examples = loaded_data
    elif num_test_samples > 0 :
        train_examples = loaded_data[:-num_test_samples]
        test_examples = loaded_data[-num_test_samples:]
    else: # No data
        train_examples = []
        test_examples = []


    print(f"Training examples: {len(train_examples)}, Test examples: {len(test_examples)}")
    

    # Begin training
    game = PokerGame() # Max_round doesn't matter here as game is for network structure
    args = dotdict({
        # Fixed hyperparams for NNet
        'lr': .0001, 
        'epochs': 8,
        'batch_size': 512, 
        'cuda': False, # Set to True if CUDA is available and desired
        "use_wandb": True, # Control wandb usage, set to False for this local test
        "wandb_run_name": run_name,
        "dim": 100,
        "dropout": 0.3,
        "entropy_coeff": 0.01, # Coefficient for entropy regularization
    })
    model = nn.NNetWrapper(game, args)

    class Fake: # Helper class for training, as NNetWrapper.train expects board objects
        def __init__(self, data_vector):
            self.data_vector = data_vector

        def to_vector(self): # Method to mimic PokerState.to_vector()
            return self.data_vector

    if train_examples:
        print("Starting training...")
        model.train([(Fake(state_vec), pi, v) for state_vec, pi, v in train_examples])
        # Save the model
        model_save_path = os.path.join(checkpoint_dir, f"{run_name}.pth.tar")
        model.save_checkpoint(folder=checkpoint_dir, filename=f"{run_name}.pth.tar")
        print(f"Model saved to {model_save_path}")
    else:
        print("No training examples. Skipping training.")

    # Test the model
    if test_examples:
        test_nn(model, [(Fake(state_vec), pi, v) for state_vec, pi, v in test_examples])
    else:
        print("No test examples. Skipping testing.")
