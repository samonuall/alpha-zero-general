import os, sys
sys.path.append(os.environ["BASE_DIR"] + "/alpha-zero-general/")
sys.path.append(os.getcwd())
print(sys.path)

import numpy as np
import math
from pypokerengine.utils.card_utils import gen_deck, estimate_hole_card_win_rate
from pypokerengine.engine.card import Card
from poker.PokerGame import PokerGame
from poker.pytorch.NNet import NNetWrapper as nn
from utils import dotdict

class PokerSearch():
    """
    This class handles the Poker search algorithm.
    It performs a one-step lookahead, and for opponent turns,
    it samples opponent hole cards and actions based on nnet policy.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.num_opponent_sims = getattr(args, 'numOpponentSims', getattr(args, 'numRandomSims', 10))

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs a search from canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   derived from the evaluated values of taking each action.
        """
        action_values = [-float('inf')] * self.game.getActionSize()
        valid_actions = self.game.getValidMoves(canonicalBoard, 1)

        player1_uuid = list(canonicalBoard.player_states.keys())[0]
        opponent_uuid = list(canonicalBoard.player_states.keys())[1]

        for action_idx in range(self.game.getActionSize()):
            if not valid_actions[action_idx]:
                continue

            s_after_player_action, player_after_player_action = self.game.getNextState(canonicalBoard, 1, action_idx)
            
            game_ended_reward = self.game.getGameEnded(s_after_player_action, 1) # Player 1's perspective

            if game_ended_reward != 0:
                action_values[action_idx] = game_ended_reward
            elif player_after_player_action == 1: # Player 1's turn again
                # Board is already canonical for player 1 as it resulted from player 1's action
                # and it's still player 1's turn.
                _, v = self.nnet.predict(s_after_player_action)
                action_values[action_idx] = v
            else: # Opponent's turn (player_after_player_action == -1)
                opponent_simulation_values = []
                for sim_idx in range(self.num_opponent_sims):
                    prev_board = canonicalBoard.copy()
                    prev_board.emulator_state["table"].deck.shuffle()
                    s_after_player_action, player_after_player_action = self.game.getNextState(prev_board, 1, action_idx)
                    
                    board_for_opp_sim = s_after_player_action.copy()
                    board_for_opp_sim.seed = canonicalBoard.seed + sim_idx # Vary seed for future card draws

                    # Randomize Opponent's Hole Cards
                    player1_hole_cards_str = board_for_opp_sim.player_states[player1_uuid].get_hole()
                    community_cards_str = board_for_opp_sim.player_states[player1_uuid].get_community()

                    player1_hole_cards_obj = [Card.from_str(c) for c in player1_hole_cards_str]
                    community_cards_obj = [Card.from_str(c) for c in community_cards_str]
                    
                    known_cards = player1_hole_cards_obj + community_cards_obj
                    
                    deck = gen_deck(exclude_cards=known_cards)
                    if len(deck.deck) < 2: # Not enough cards to deal to opponent
                        # This scenario should be rare but could happen if deck is exhausted
                        # Assign a neutral or bad value for this simulation path
                        opponent_simulation_values.append(0) # Or some other default
                        continue

                    opponent_new_holes_obj = deck.draw_cards(2)
                    opponent_new_holes_str = [str(c) for c in opponent_new_holes_obj]
                    
                    board_for_opp_sim.player_states[opponent_uuid].set_hole(opponent_new_holes_str)

                    # Update opponent's EHS
                    ehs_opponent = estimate_hole_card_win_rate(
                        nb_simulation=100, # Standard number of simulations for EHS
                        nb_player=2,
                        hole_card=opponent_new_holes_obj,
                        community_card=community_cards_obj if community_cards_obj else None
                    )
                    board_for_opp_sim.player_states[opponent_uuid]["EHS"] = ehs_opponent

                    # Get opponent's canonical board and predict action
                    s_opp_canonical = self.game.getCanonicalForm(board_for_opp_sim, -1) # Opponent's perspective
                    pi_opponent, _ = self.nnet.predict(s_opp_canonical)
                    valid_moves_opponent = self.game.getValidMoves(s_opp_canonical, 1) # Moves for current player in s_opp_canonical

                    pi_opponent = pi_opponent * valid_moves_opponent
                    sum_pi_opponent = np.sum(pi_opponent)

                    if sum_pi_opponent > 0:
                        pi_opponent /= sum_pi_opponent
                    else:
                        if np.sum(valid_moves_opponent) > 0:
                            pi_opponent = valid_moves_opponent / np.sum(valid_moves_opponent)
                        else: # No valid moves for opponent, highly unlikely unless game ended
                              # This path might be terminal or problematic.
                            opponent_simulation_values.append(-1) # Penalize
                            continue
                    
                    a_opponent = np.random.choice(len(pi_opponent), p=pi_opponent)

                    # Get state after opponent's action
                    # s_opp_canonical is from opponent's view (player -1), action taken by player 1 in that view
                    s_after_opponent_action, player_idx_in_s_opp_frame = self.game.getNextState(s_opp_canonical, 1, a_opponent)

                    # Evaluate the resulting state from original player 1's perspective
                    final_value_from_p1_perspective = self.game.getGameEnded(s_after_opponent_action, 1)

                    if final_value_from_p1_perspective == 0: # Game not ended
                        # Determine whose turn it is in the original frame of reference
                        # If player_idx_in_s_opp_frame is 1, it's still the opponent's turn (relative to s_opp_canonical's player 1)
                        # So, in original frame, it's opponent's turn (-1).
                        # If player_idx_in_s_opp_frame is -1, it's original player 1's turn (relative to s_opp_canonical's player 1)
                        # So, in original frame, it's player 1's turn (1).
                        player_in_original_frame = -1 * player_idx_in_s_opp_frame
                        
                        s_final_canonical_for_eval = self.game.getCanonicalForm(s_after_opponent_action, player_in_original_frame)
                        _, v_raw = self.nnet.predict(s_final_canonical_for_eval)
                        final_value_from_p1_perspective = v_raw
                    
                    opponent_simulation_values.append(final_value_from_p1_perspective)

                if opponent_simulation_values:
                    action_values[action_idx] = np.mean(opponent_simulation_values)
                else:
                    action_values[action_idx] = -1 # Default if all opponent sims failed

        # Convert action values to probabilities
        probs = [0] * self.game.getActionSize()
        if temp == 0:
            # Filter out -inf values before finding max, only consider valid actions
            valid_action_indices = [i for i, v_a in enumerate(valid_actions) if v_a == 1]
            if not valid_action_indices: # Should not happen if game is not over
                return [1.0/len(probs) if p == 1 else 0 for p in valid_actions] # fallback, though problematic

            best_val = -float('inf')
            best_actions = []
            for i in valid_action_indices:
                if action_values[i] > best_val:
                    best_val = action_values[i]
                    best_actions = [i]
                elif action_values[i] == best_val:
                    best_actions.append(i)
            
            if best_actions:
                best_a = np.random.choice(best_actions)
                probs[best_a] = 1
            else: # Fallback if no best action found (e.g. all values are -inf)
                 # Distribute probability among valid actions
                num_valid = sum(valid_actions)
                if num_valid > 0:
                    for i in range(len(probs)):
                        if valid_actions[i]:
                            probs[i] = 1.0 / num_valid
                else: # No valid actions at all, highly problematic
                    probs = [1.0/len(probs)] * len(probs)


            return probs

        # Softmax for temp > 0
        # Only apply softmax over valid actions
        logits = []
        logit_indices = []
        for i in range(self.game.getActionSize()):
            if valid_actions[i]:
                logits.append(action_values[i] / temp)
                logit_indices.append(i)
        
        if not logits: # No valid actions with values
            num_valid = sum(valid_actions)
            if num_valid > 0:
                for i in range(len(probs)):
                    if valid_actions[i]:
                        probs[i] = 1.0 / num_valid
            else: # No valid actions at all
                 probs = [1.0/len(probs)] * len(probs)
            return probs


        exp_logits = np.exp(np.array(logits) - np.max(logits)) # Subtract max for stability
        sum_exp_logits = np.sum(exp_logits)

        if sum_exp_logits > 0:
            probabilities = exp_logits / sum_exp_logits
            for i, prob_val in enumerate(probabilities):
                probs[logit_indices[i]] = prob_val
        else: # All exp_logits are zero (e.g. all logits were -inf or extremely small)
            # Uniform distribution among valid actions
            num_valid_logits = len(logits)
            if num_valid_logits > 0:
                for idx in logit_indices:
                    probs[idx] = 1.0 / num_valid_logits
            else: # Should be caught by "if not logits"
                num_valid_total = sum(valid_actions)
                if num_valid_total > 0:
                    for i in range(len(probs)):
                        if valid_actions[i]:
                            probs[i] = 1.0 / num_valid_total
                else:
                    probs = [1.0/len(probs)] * len(probs)
        
        return probs

if __name__ == "__main__":
    # Example usage
    args = dotdict({
        'numOpponentSims': 20,
        'numRandomSims': 20,
        'numMCTSSims': 2,
        'cpuct': 1.0,
        'temperature': 1.0,
        "dim": 100,
        "dropout": 0.3,
        "use_wandb": False,
        "cuda": False
    })
    game = PokerGame(max_round=1)
    nnet = nn(game, args)
    nnet.load_checkpoint(folder="alpha-zero-general/pretrained_data", filename="2_000data_naiveplayer_50rounds.pth.tar")
    search = PokerSearch(game, nnet, args)
    board = game.getInitBoard()
    board = game.getCanonicalForm(board, 1)
    board, currPlayer = game.getNextState(board, 1, 0) # Example action

    board = game.getCanonicalForm(board, currPlayer)
    probs = search.getActionProb(board, temp=0)
    print("Action probabilities:", probs)