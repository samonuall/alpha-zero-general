import sys, os
sys.path.append(os.getcwd())

import numpy as np
import math
from pypokerengine.utils.game_state_utils import deepcopy_game_state

rank_converter = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12
}

suit_converter = {
    "C": 0,
    "D": 1,
    "H": 2,
    "S": 3
}

street_converter = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,

}

rank_rev_converter = {v: k for k, v in rank_converter.items()}
suit_rev_converter = {v: k for k, v in suit_converter.items()}
street_rev_converter = {v: k for k, v in street_converter.items()}
move_rev_converter = {0: "raise", 1: "call", 2: "check"}

class PokerBoard:
    def __init__(self, players_info):
        self.player_states = {uuid: PokerState() for uuid in players_info}
        self.emulator_state = None
        self.finished = False
        self.winner = None
        self.end_stacks = {uuid: 0 for uuid in players_info}
        self.seed = None

    def to_vector(self):
        # Convert player states of whoever player 1 is to vector
        uuids = list(self.player_states.keys())
        return self.player_states[uuids[0]].to_vector()
        
    def to_string(self):
        # Always prints out perspective of player 1, not player -1
        uuids = list(self.player_states.keys())
        return str(self.finished) + "\n" +  str(self.player_states[uuids[0]])
    
    def copy(self):
        # Create a deep copy of the board
        new_board = PokerBoard(self.player_states.keys())
        new_states = {}
        for uuid, state in self.player_states.items():
            new_states[uuid] = PokerState()
            new_states[uuid].features = state.features.copy()
            new_states[uuid].hole_cards = state.hole_cards.copy()
            new_states[uuid].community_cards = state.community_cards.copy()
        new_board.player_states = new_states
        new_board.finished = self.finished
        new_board.winner = self.winner
        new_board.end_stacks = self.end_stacks.copy()
        new_board.seed = self.seed
        new_board.emulator_state = deepcopy_game_state(self.emulator_state) if self.emulator_state else None
        return new_board


# Action will be: [raise, call, check, fold]
# Use an emulator and create a player object with the neural network.
class PokerState:
    """
    Holds features we will use for poker. Basically a dictionary that initializes the features
    we want and makes sure that the structure cannot be changed. This is used to represent state for the neural network.
    There is a separate game state that the emulator will use."""
    def __init__(self):
        # Use direct assignment to self.__dict__ to avoid triggering __setattr__
        self.__dict__['features'] = {
            'EHS': 0.0,
            "round_count": 0.0,
            'community_suit': np.zeros(4, dtype=int),
            'community_rank': np.zeros(13, dtype=int),
            'hole_suit_1': np.zeros(4, dtype=int),
            'hole_rank_1': np.zeros(13, dtype=int),
            'hole_suit_2': np.zeros(4, dtype=int),
            'hole_rank_2': np.zeros(13, dtype=int),
            'pot': 0,
            'street': np.zeros(4, dtype=int),
            'pot_odds': 0.0,
            'my_stack': 0.0,
            'opp_stack': 0.0,
            'num_raises_this_street': 0.0
        }
        self.hole_cards = []
        self.community_cards = []
    
    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, value):
        assert type(self.features[key]) == type(value), "New value must be same type"

        if isinstance(value, np.ndarray):
            assert value.shape == self.features[key].shape, "Must be same shape as original"
        
        self.features[key] = value

    def __contains__(self, key):
        return key in self.features
    
    def set_pot_odds(self, pot, price):
        price /= 20 # Scale by big blind amount
        pot /= 20 # Scale by big blind amount
        self.features["pot_odds"] = price / (price + pot)
    
    def set_stack(self, my_stack, opp_stack):
        # Scale by big blind amount
        my_stack /= 20
        opp_stack /= 20
        self.features["my_stack"] = float(my_stack)
        self.features["opp_stack"] = float(opp_stack)
    
    def set_num_raises_this_street(self, num_raises):
        # Scale by big blind amount
        self.features["num_raises_this_street"] = float(num_raises)
    
    def get_hole(self):
        return self.hole_cards
    
    def set_hole(self, hole_cards):
        self.hole_cards = hole_cards
        
        # Reset hole features
        self.features["hole_suit_1"] = np.zeros(4, dtype=int)
        self.features["hole_rank_1"] = np.zeros(13, dtype=int)
        self.features["hole_suit_2"] = np.zeros(4, dtype=int)
        self.features["hole_rank_2"] = np.zeros(13, dtype=int)

        # Add hole cards
        self.features["hole_suit_1"][suit_converter[hole_cards[0][0]]] = 1
        self.features["hole_rank_1"][rank_converter[hole_cards[0][1]]] = 1
        self.features["hole_suit_2"][suit_converter[hole_cards[1][0]]] = 1
        self.features["hole_rank_2"][rank_converter[hole_cards[1][1]]] = 1
    
    def set_community(self, community_cards):
        self.community_cards = community_cards
        
        # Reset community features
        self.features["community_suit"] = np.zeros(4, dtype=int)
        self.features["community_rank"] = np.zeros(13, dtype=int)
        
        # Add community cards
        for card in community_cards:
            self.features["community_suit"][suit_converter[card[0]]] += 1
            self.features["community_rank"][rank_converter[card[1]]] += 1

    def get_community(self):
        return self.community_cards
    
    def set_pot(self, pot, big_blind=20):
        # Scale by raise amount
        pot /= big_blind
        self.features["pot"] = float(pot)
    
    def set_street(self, street):
        """
        PREFLOP = 0
        FLOP = 1
        TURN = 2
        RIVER = 3
        SHOWDOWN = 4
        FINISHED = 5
        """
        # Reset street features
        self.features["street"] = np.zeros(4, dtype=int)
        if street < 4:
            self.features["street"][street] = 1

    def set_round_count(self, round_count):
        bucket = math.floor(round_count / 5) * 5
        self.features["round_count"] = float(bucket)
    
    def clear(self):
        # Clear values
        self.features = {
            'EHS': 0.0,
            "stack": 0.0, # Normalize like pot based on raise amount
            'community_suit': np.zeros(4, dtype=int),
            'community_rank': np.zeros(13, dtype=int),
            'hole_suit_1': np.zeros(4, dtype=int),
            'hole_rank_1': np.zeros(13, dtype=int),
            'hole_suit_2': np.zeros(4, dtype=int),
            'hole_rank_2': np.zeros(13, dtype=int),
            'pot': 0.0,
            'street': np.zeros(4, dtype=int),
            'pot_odds': 0.0,
            'my_stack': 0.0,
            'opp_stack': 0.0,
            'num_raises_this_street': 0.0
        }
        
    
    def to_vector(self):
        # Convert all features to a single numpy vector
        vectors = []
        for value in self.features.values():
            if isinstance(value, np.ndarray):
                vectors.append(value)
            else:
                vectors.append(np.array([value]))
        
        # Concatenate all feature vectors into a single vector
        feature_vector = np.concatenate(vectors)
        return feature_vector
    
    def __str__(self):
        """Bucket some values so that during MCTS search we reduce the search space"""
        result = []
        for key, value in self.features.items():
            if key == 'EHS':
                readable_value = ""
            elif key == 'pot':
                readable_value = str(int(self.features['pot']))
            elif key == 'round_count':
                readable_value = str(int(self.features['round_count']))
            elif key == 'community_suit':
                readable_value = ""
            elif key == 'community_rank':
                readable_value = ""
            elif key == 'hole_suit_1':
                readable_value = ""
            elif key == 'hole_rank_1':
                readable_value = ""
            elif key == 'hole_suit_2':
                readable_value = ""
            elif key == 'hole_rank_2':
                readable_value = ""
            elif key == 'pot_odds':
                readable_value = ""
            elif key == 'street':
                street_name = [street_rev_converter[i] for i, present in enumerate(value) if present]
                readable_value = street_name[0] if street_name else "None"
            else:
                 readable_value = str(value) # Default for any other keys

            result.append(f"{key}: {readable_value}")

        # Sort cards so that states with same cards are the same
        hole_str = ", ".join(sorted(self.hole_cards)) if self.hole_cards else "None"
        comm_str = ", ".join(sorted(self.community_cards)) if self.community_cards else "None"
        result.insert(0, f"Hole Cards: {hole_str}")
        result.insert(1, f"Community Cards: {comm_str}")

        return "\n".join(result)
    



if __name__ == "__main__":
    # Test some stuff
    features = PokerState()
    hole_cards = ['HA', 'D5']
    community_cards = ['DK', 'SK', 'C8', 'S2', 'H8']
    pot = 50
    street = 0
    round_count = 5.0

    features.set_hole(hole_cards)
    features.set_community(community_cards)
    features.set_pot(50)
    features.set_street(street)
    features.set_round_count(round_count)
    features.set_stack(1000, 1000)
    features.set_num_raises_this_street(2)
    features.set_pot_odds(50, 20)


    print(features.to_vector())
    print()
    print(features)