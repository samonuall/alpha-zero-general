import numpy as np
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.utils.card_utils import gen_cards
from MCTS import MCTS

class NNetPlayer():
    def __init__(self, game, nnet, args):
        self.mcts = MCTS(game, nnet, args)

    def play(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=1))


class NaivePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        uuids = list(board.player_states.keys())
        hole_card = board.player_states[uuids[0]].hole_cards
        round_state = DataEncoder.encode_round_state(board.emulator_state)
        valid_actions = self.game.getValidMoves(board, 1)
        
        win_prob = estimate_hole_card_win_rate(300, 2, gen_cards(hole_card), gen_cards(round_state["community_card"]))
        if win_prob >= 0.7 and valid_actions[0] == 1:
            # raise
            action = 0
        elif win_prob > 0.3 and valid_actions[1] == 1 or valid_actions[2] == 1: 
            # check or call
            if valid_actions[1] == 1:
                action = 1
            elif valid_actions[2] == 1:
                # fold
                action = 2
        elif valid_actions[3] == 1:
            action = 3
        else:
            # pick whatever is valid
            action = valid_actions.index(1)

        return action


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        # Filter out invalid moves
        valid_indices = [i for i, valid in enumerate(valids) if valid]
        # Choose a random action from the valid ones
        a = np.random.choice(valid_indices)
        return a

class HumanPokerPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        print("Round state:")
        print(DataEncoder.encode_round_state(board.emulator_state))
        # display(board) # Assuming a display function exists or is added later
        print("Your board state:")
        uuids = list(board.player_states.keys())
        print(f"Your UUID: {uuids[0]}")
        print(board.player_states[uuids[0]])
        print()
        valid = self.game.getValidMoves(board, 1)
        action_map = {0: "raise", 1: "call", 2: "check", 3: "fold"}
        print("Valid moves:")
        for i in range(len(valid)):
            if valid[i]:
                print(f"[{i}: {action_map[i]}] ", end="")
        print() # Newline after printing moves

        while True:
            try:
                input_move = input("Enter your move index: ")
                a = int(input_move)
                if 0 <= a < self.game.getActionSize() and valid[a]:
                    break
                else:
                    print('Invalid move index or action not allowed.')
            except ValueError:
                print('Invalid input. Please enter an integer index.')
            except EOFError: # Handle cases where input stream is closed (e.g., in automated testing)
                print('Input stream closed. Choosing first valid move.')
                valid_indices = [i for i, v in enumerate(valid) if v]
                if valid_indices:
                    a = valid_indices[0]
                    break
                else: # Should not happen if game logic is correct
                    raise Exception("No valid moves available for Human player.")

        return a

class GreedyPokerPlayer():
    """
    A simple greedy player: Prioritizes Raise > Call > Check > Fold.
    """
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        # Action indices: 0: raise, 1: call, 2: check, 3: fold
        if valids[0]: # If raise is valid
            return 0
        elif valids[1]: # If call is valid
            return 1
        elif valids[2]: # If check is valid
            return 2
        elif valids[3]: # If fold is valid
            return 3
        else:
            # This should theoretically not be reached if the game provides valid moves correctly
            raise Exception("No valid moves found for Greedy player.")

