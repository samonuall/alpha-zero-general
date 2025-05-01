import sys
import os

# Add the parent directory of both alpha-poker and pypokerengine to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
alpha_poker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if alpha_poker_dir not in sys.path:
    sys.path.insert(0, alpha_poker_dir)

from Game import Game
from pypokerengine.api.emulator import Emulator, Event
from poker.PokerLogic import PokerBoard, PokerState
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.engine.card import Card
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.action_checker import ActionChecker
import random
import time
from contextlib import contextmanager
import numpy as np
from pypokerengine.engine.poker_constants import PokerConstants as Const
from poker.boardtexture import classify_board_texture, board_texture_to_onehot # Added import

# Sets the seed so that shuffling is deterministic
@contextmanager
def seeded(seed):
    prev = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(prev)

class FakePlayer:
    def __init__(self):
        self.uuid = None

# Game ended is where you return reward, so it doesn't really have to be -1 or 1. Might have to change code tho
# From a quick look probably can do r > 1, but we'll see

# Board for the poker game includes the state returned by the emulator + events. I need to look in the Coach.py to figure out how it stores
# the features for training the neural network to make sure we use PokerState.to_vector() at the right time to make this work.
# could end up being that the board is actually PokerState and we just have to figure out how to turn emulator state and events into this.
# Board should be more complicated than pokerstate, it will include everything including events and game state to make sure you can pass it in.

# We could make a game last multiple rounds (maybe should)
# TODO: Make sure states are properly represented, then pit against random and raise player to see if it learns after a couple epochs
def init_emulator(players_info, max_round=10):
    emulator = Emulator()
    
    emulator.set_game_rule(
        player_num = 2,
        max_round = max_round,
        small_blind_amount = 10,
        ante_amount = 0,
    )

    uuids = list(players_info.keys())
    
    # TODO: change to work with a dummy player instead of a real player
    # register first player
    player = FakePlayer()
    player.uuid = uuids[0]
    emulator.register_player(
        uuid = player.uuid,
        player = player
    )

    # register second player
    player = FakePlayer()
    player.uuid = uuids[1]
    emulator.register_player(
        uuid = player.uuid,
        player = player
    )

    return emulator


def get_last_n_actions(action_history, player_uuid, n, uuid_map=None):
        """
        Returns the last n actions (as strings) taken by the opponent to player_uuid, ordered oldest to newest.
        action_history: dict of street -> list of action dicts (see example above)
        player_uuid: string
        n: int
        """
        print("ACTION HISTORY:", action_history)
        actions = []
        
        # Flatten all actions in order of streets
        street_order = ['preflop', 'flop', 'turn', 'river']
        all_actions = []
        for street in street_order:
            if street in action_history:
                all_actions.extend(action_history[street])
        
        # Filter for actions by the other player
        for act in all_actions:
            if act.get('uuid') != player_uuid:
                if uuid_map:
                    if act.get('uuid') == uuid_map[player_uuid]:
                        continue
                action = act["action"].lower()
                if action == "bigblind" or action == "smallblind":
                    continue
                elif action == "call" and act["amount"] == 0:
                    action = "check"
                actions.append(action)
        
        # Return the last n, oldest to newest (left oldest, right newest)
        return actions[-n:] if n > 0 else []


MAX_ROUND = 10
class PokerGame(Game):
    def __init__(self, seed=None, max_round=MAX_ROUND, uuids=None):
        super().__init__()
        # players info is how each round will start
        self.players_info = {
            "player1": {"stack": 1000, "name": "Alice"},
            "player2": {"stack": 1000, "name": "Bob"}
        }
        self.emulator = init_emulator(self.players_info, max_round=MAX_ROUND)
        self.max_round = max_round
        self.seed = seed
        if uuids:
            self.uuid_map = {player: uuid for uuid, player in zip(uuids, self.emulator.players_holder)}
        else:
            self.uuid_map = None

    def getInitBoard(self):
        board = PokerBoard(self.players_info)

        # Create 32 bit random seed so that all future states are deterministic from this start
        if self.seed is None:
            # Random games
            board.seed = random.getrandbits(32)
        else:
            # we want board to be deterministic
            board.seed = self.seed
            self.seed += 1 # add one so that next board in this game object is different in deterministic way
    
        with seeded(board.seed):
            # Get initial state, make player 1 actually first with this swap of players info
            players_info = {
                "player2": {"stack": 1000, "name": "Bob"},
                "player1": {"stack": 1000, "name": "Alice"}
            }
            initial_game_state = self.emulator.generate_initial_game_state(players_info)

            # Start round
            board.emulator_state, events = self.emulator.start_new_round(initial_game_state)
        
        # Update poker states in board
        round_state = DataEncoder.encode_round_state(board.emulator_state)
        
        # Calculate initial board texture (preflop has no community cards, handle this)
        community_cards = round_state["community_card"]
        if len(community_cards) >= 3:
            texture_dict = classify_board_texture(board.emulator_state["street"], (), community_cards)
            texture_onehot = board_texture_to_onehot(texture_dict)
        else: # Preflop case
            texture_onehot = np.zeros(7, dtype=np.int32) # Default zero vector for preflop

        for i, player_uuid in enumerate(self.emulator.players_holder):
            hole_card = board.emulator_state["table"].seats.players[i].hole_card
            ehs = estimate_hole_card_win_rate(
                100, 
                2, 
                hole_card, 
                [Card.from_str(card) for card in community_cards]
            )
            
            board.player_states[player_uuid]["EHS"] = ehs
            board.player_states[player_uuid].set_round_count(1)
            board.player_states[player_uuid].set_community(community_cards)
            board.player_states[player_uuid].set_hole(list(map(str, hole_card)))
            board.player_states[player_uuid].set_pot(round_state["pot"]["main"]["amount"])
            board.player_states[player_uuid].set_street(board.emulator_state["street"])
            board.player_states[player_uuid].set_board_texture(texture_onehot) # Set initial board texture

            # Set stacks
            stacks = [board.emulator_state["table"].seats.players[i].stack for i in range(2)]
            if i == 0:
                board.player_states[player_uuid].set_stack(stacks[1], stacks[0])
            else:
                board.player_states[player_uuid].set_stack(stacks[0], stacks[1])

        return board

    def getBoardSize(self):
        return 73

    def getActionSize(self):
        return 4
    
    # player 1 corresponds with positon 1, and player -1 corresponds with position 0
    def getNextState(self, board, player, action):
        """Action is integer index for [raise, call, check, fold]"""
        # Preserves order of uuids
        board = board.copy()
        result_board = PokerBoard(list(board.player_states.keys()))

        actions = ["raise", "call", "check", "fold"]
        if action == 2:
            action = "call"
        else:
            action = actions[action]
        
        result_board.emulator_state, events = self.emulator.apply_action(board.emulator_state, action)
        
        street_change = False
        round_change = False
        # handle events
        for event in events:
            if event["type"] == "event_new_street":
                # Recalculate EHS
                street_change = True
            
            elif event["type"] == "event_round_finish":
                # increment round count, start new round
                board.seed += 1 # change deterministically
                with seeded(board.seed):
                    new_emulator_state, new_round_events = self.emulator.start_new_round(result_board.emulator_state)
                round_change = True

                if self.max_round + 1 == new_emulator_state["round_count"]:
                    board.finished = True
                    # set final player stacks
                    round_winner_uuid = event["winners"][0]["uuid"]
                    round_winner_stack = event["winners"][0]["stack"]
                    board.end_stacks[round_winner_uuid] = round_winner_stack
                    uuids = list(board.player_states.keys())
                    if round_winner_uuid == uuids[0]:
                        other_uuid = uuids[1]
                    else:
                        other_uuid = uuids[0]
                    board.end_stacks[other_uuid] = 2000 - round_winner_stack

                    # Figure out winner from stacks
                    if round_winner_stack > board.end_stacks[other_uuid]:
                        board.winner = round_winner_uuid
                    elif round_winner_stack < board.end_stacks[other_uuid]:
                        board.winner = other_uuid

                    board.emulator_state = result_board.emulator_state
                    return board, -player

                
                elif Event.GAME_FINISH == new_round_events[-1]["type"]:
                    # use input board since you haven't updated the other one yet
                    board.finished = True
                    # calculate winner
                    winner = None
                    max_stack = 0
                    for player_info in new_round_events[-1]["players"]:
                        board.end_stacks[player_info["uuid"]] = player_info["stack"]
                        if player_info["stack"] > max_stack:
                            max_stack = player_info["stack"]
                            winner = player_info["uuid"]
                    board.winner = winner
                    board.emulator_state = result_board.emulator_state
                    return board, -player
                
                else:
                    result_board.emulator_state = new_emulator_state
            
            elif event["type"] == "event_game_finish":
                board.finished = True
                # calculate winner
                winner = None
                max_stack = 0
                for player_info in event["players"]:
                    board.end_stacks[player_info["uuid"]] = player_info["stack"]
                    if player_info["stack"] > max_stack:
                        max_stack = player_info["stack"]
                        winner = player_info["uuid"]
                board.winner = winner
                board.emulator_state = result_board.emulator_state
                return board, -player

        # Get player positions from emulator point of view, this is important
        uuids = list(board.player_states.keys())
        if player == 1:
            player_pos = 1 if uuids[0] == "player1" else 0
        else:
            player_pos = 0 if uuids[0] == "player1" else 1
        
        # set current player states
        round_state = DataEncoder.encode_round_state(result_board.emulator_state)
        for player_uuid in board.player_states.keys():
            if round_change:
                # New round number
                result_board.player_states[player_uuid].set_round_count(float(result_board.emulator_state["round_count"]))
                # New hole
                index = 0 if player_uuid == "player1" else 1
                hole_card = result_board.emulator_state["table"].seats.players[index].hole_card
                result_board.player_states[player_uuid].set_hole(list(map(str, hole_card)))
                # New EHS
                result_board.player_states[player_uuid]["EHS"] = estimate_hole_card_win_rate(
                    100, 
                    2, 
                    hole_card, 
                    [Card.from_str(card) for card in round_state["community_card"]]
                )
                # New pot
                result_board.player_states[player_uuid].set_pot(round_state["pot"]["main"]["amount"])
                # New community cards
                community_cards = round_state["community_card"]
                result_board.player_states[player_uuid].set_community(community_cards)
                result_board.player_states[player_uuid].set_street(result_board.emulator_state["street"])
                # New board texture (preflop case)
                texture_onehot = np.zeros(7, dtype=np.int32) # Default zero vector for preflop
                result_board.player_states[player_uuid].set_board_texture(texture_onehot)
            
            elif street_change and result_board.emulator_state["street"] != "showdown":
                # Community cards have most likely changed
                hole_cards = board.player_states[player_uuid].get_hole()
                result_board.player_states[player_uuid]["EHS"] = estimate_hole_card_win_rate(
                    100, 
                    2, 
                    [Card.from_str(card) for card in hole_cards], 
                    [Card.from_str(card) for card in round_state["community_card"]]
                )
                community_cards = round_state["community_card"]
                result_board.player_states[player_uuid].set_community(community_cards)
                
                # Calculate new board texture
                texture_dict = classify_board_texture(result_board.emulator_state["street"], (), community_cards)
                texture_onehot = board_texture_to_onehot(texture_dict)
                result_board.player_states[player_uuid].set_board_texture(texture_onehot)

                # everything else same
                result_board.player_states[player_uuid].set_round_count(float(board.emulator_state["round_count"]))
                result_board.player_states[player_uuid].set_hole(board.player_states[player_uuid].get_hole())
                result_board.player_states[player_uuid].set_pot(round_state["pot"]["main"]["amount"])
                result_board.player_states[player_uuid].set_street(result_board.emulator_state["street"])
            
            else:
                # Most is the same except for pot and opponent moves
                result_board.player_states[player_uuid]["EHS"] = board.player_states[player_uuid]["EHS"]
                result_board.player_states[player_uuid].set_community(board.player_states[player_uuid].get_community())
                result_board.player_states[player_uuid].set_round_count(float(board.emulator_state["round_count"]))
                result_board.player_states[player_uuid].set_hole(board.player_states[player_uuid].get_hole())
                result_board.player_states[player_uuid].set_pot(round_state["pot"]["main"]["amount"])
                result_board.player_states[player_uuid].set_street(result_board.emulator_state["street"])
                # Pass board texture along
                result_board.player_states[player_uuid].set_board_texture(board.player_states[player_uuid].get_board_texture())
        
            # Set stacks
            players = result_board.emulator_state["table"].seats.players
            for i in range(2):
                curr_player = players[i]
                other_player = players[1 - i]
                result_board.player_states[curr_player.uuid].set_stack(curr_player.stack, other_player.stack)

            # Set pot odds
            if result_board.emulator_state["street"] < 2:
                price = 20
            else:
                price = 40
            result_board.player_states[player_uuid].set_pot_odds(round_state["pot"]["main"]["amount"], price)

            # Set number of raises
            if street_change:
                num_raises_this_street = 0
            elif action == "raise":
                num_raises_this_street = board.player_states[player_uuid]["num_raises_this_street"] + 1
            else:
                num_raises_this_street = board.player_states[player_uuid]["num_raises_this_street"]
            
            result_board.player_states[player_uuid].set_num_raises_this_street(num_raises_this_street)
        
        result_board.end_stacks = board.end_stacks.copy()
        result_board.finished = board.finished
        result_board.winner = board.winner
        result_board.seed = board.seed

        if list(board.player_states.keys())[0] == "player1":
            next_player = 1 if result_board.emulator_state["next_player"] == 1 else -1
        else:
            # Swappped
            next_player = 1 if result_board.emulator_state["next_player"] == 0 else -1
        
        return result_board, next_player

    def getValidMoves(self, board, player):
        board = board.copy()
        # Needs to take into account number of raises allowed
        players = board.emulator_state["table"].seats.players

        # Kind of complex, just making sure that if the player perspectives were swapped the actions are still valid
        uuids = list(board.player_states.keys())
        if player == 1:
            player_pos = 1 if uuids[0] == "player1" else 0
        else:
            player_pos = 0 if uuids[0] == "player1" else 1

        # DEBUG: Check if player_pos is correct
        if player == 1:
            expected_emulator_player = 1 if list(board.player_states.keys())[0] == "player1" else 0
            assert board.emulator_state['next_player'] == expected_emulator_player, \
                f"Emulator next_player {board.emulator_state['next_player']} mismatch! Expected {expected_emulator_player} for canonical player 1 with keys {list(board.player_states.keys())}"
            assert player_pos == board.emulator_state['next_player'], \
                f"Calculated player_pos {player_pos} != emulator next_player {board.emulator_state['next_player']} for board keys {list(board.player_states.keys())}"
        
        sb_amount = board.emulator_state["small_blind_amount"]
        legal_actions = ActionChecker.legal_actions(players, player_pos, sb_amount, board.emulator_state["street"])

        # Order is ["raise", "call", "check", "fold"]
        result = [0] * self.getActionSize()
        for action in legal_actions:
            if action["action"] == "raise":
                result[0] = 1
            
            elif action["action"] == "call":
                # Need to find last opponent move to see if a call is a check or call
                uuids = list(board.player_states.keys())
                if player == 1:
                    uuid = uuids[0]
                else:
                    uuid = uuids[1]
                
                round_state = DataEncoder.encode_round_state(board.emulator_state)
                opponent_moves = get_last_n_actions(round_state["action_histories"], uuid, 1, uuid_map=self.uuid_map)

                # Figure out if opponent's last move was a raise or not
                raise_move = (len(opponent_moves) > 0 and opponent_moves[-1] == "raise")
                
                # figure out if player is at start of new street, if so check is valid
                streets = ["preflop", "flop", "turn", "river"]
                if raise_move and len(round_state["action_histories"][streets[board.emulator_state["street"]]]) > 0:
                    # If they raised, you can't check you must call
                    result[1] = 1
                else:
                    # If they checked, called, or folded you can also check
                    result[2] = 1
            
            elif action["action"] == "fold":
                result[3] = 1

        if sum(result) == 0:
            print("-----NO VALID ACTIONS------")
        return result

    # TODO: change to return pot instead of 1 and -1, and change coach code to work with non-binary rewards
    def getGameEnded(self, board, player):
        board = board.copy()
        # Return whatever reward seems right, I'll figure out later what I need to change to make it work
        if board.finished:
            uuids = list(board.player_states.keys())
            if player == 1:
                curr_player = uuids[0]
                opponent = uuids[1]
            else:
                curr_player = uuids[1]
                opponent = uuids[0]

            scaled_reward = (board.end_stacks[curr_player] - 1000) / 1000
            print("ENDSTACKS:", board.end_stacks)
            if board.winner == None:
                return 0.001
            elif board.winner == curr_player:
                return scaled_reward
            elif board.winner == opponent:
                return scaled_reward
            else:
                raise Exception("Winner not in player states")
        
        else:
            return 0
    
    def getCanonicalForm(self, board, player):
        """Return poker state from the perspective of teh input player. Probably just means switching the hole cards around."""
        # Canocial board also must return something that can be passed right into the neural network
        # create copy of board
        if player == 1:
            return board
        else:
            # Swap order of states in player_states
            new_board = PokerBoard(list(board.player_states.keys()))
            new_board.emulator_state = board.emulator_state.copy()
            new_board.finished = board.finished
            new_board.winner = board.winner
            new_board.end_stacks = board.end_stacks.copy()
            new_board.seed = board.seed
            # Swap player states
            new_player_states = {}
            for uuid in reversed(board.player_states.keys()):
                new_player_states[uuid] = PokerState()
                new_player_states[uuid].features = board.player_states[uuid].features.copy()
                new_player_states[uuid].hole_cards = board.player_states[uuid].hole_cards.copy()
                new_player_states[uuid].community_cards = board.player_states[uuid].community_cards.copy()
            
            new_board.player_states = new_player_states
            return new_board

    def getSymmetries(self, board, pi):
        # No symmetries for poker. NNet will call the to_vector function for the board during training
        return [(board, pi)]
    
    def stringRepresentation(self, board):
        # String representation is used only for canonical boards, meaning you should only include information from one player's point of view
        return board.to_string()
    

if __name__ == "__main__":
    round_history = {'preflop': [{'action': 'SMALLBLIND', 'amount': 10, 'add_amount': 10, 'uuid': 'player2'}, {'action': 'BIGBLIND', 'amount': 20, 'add_amount': 10, 'uuid': 'player1'}, {'action': 'CALL', 'amount': 20, 'paid': 10, 'uuid': 'player2'}, {'action': 'CALL', 'amount': 20, 'paid': 0, 'uuid': 'player1'}], 'flop': [{'action': 'RAISE', 'amount': 20, 'paid': 20, 'add_amount': 20, 'uuid': 'player2'}]}

    game = PokerGame()
    start_board = game.getInitBoard()
    curr_player = 1
    start_board = game.getCanonicalForm(start_board, 1)
    encoded_start_board = DataEncoder.encode_round_state(start_board.emulator_state)
    board = start_board
    print("Start board:", encoded_start_board)
    
    train_examples = [] # [board, pi, value]
    while game.getGameEnded(board, curr_player) == 0:
        # do random action from legal actions
        legal_actions = game.getValidMoves(board, 1)
        action = random.choice([i for i, x in enumerate(legal_actions) if x == 1 and i != 3])
        actions = ["raise", "call", "check", "fold"]
        
        print()
        sym, pi = game.getSymmetries(board, [.25] * 4)[0]
        # simulate a game where winning player wins 500
        train_examples.append([sym, pi, .5 * curr_player])

        print("Taking action:", actions[action])
        board, curr_player = game.getNextState(board, 1, action)
        board = game.getCanonicalForm(board, curr_player)
        print(board.to_string())
        print("-" * 50)
        
    print("\n\n")
    print("Winner:", board.winner)
    print("End stacks:", board.end_stacks)
    print(game.getGameEnded(board, curr_player))
    print()
    print(game.stringRepresentation(board))
    print(board.emulator_state)

    # from poker.pytorch.NNet import NNetWrapper
    # from utils import dotdict
    # args = dotdict({
    #     'lr': .0005,
    #     'dropout': 0.3,
    #     'epochs': 10,
    #     'batch_size': 64,
    #     'cuda': False,
    #     'block_width': 256,
    #     'n_blocks': 1,
    #     "wandb_run_name": "test_run"
    # })
    # nnet = NNetWrapper(game, args)
    # nnet.train(train_examples * 10)
    # print(nnet.predict(board))

    # game = PokerGame()
    # start_board = game.getInitBoard()  
    # curr_player = 1
    # start_board = game.getCanonicalForm(start_board, 1)

    # TODO: Something very wrong, running from same initial state somehow ends up with different hole cards, emulator state may carry over
    # Or its randomizing somewhere it shouldn't
    # Thing that doesnt make sense is hole cards are different between states??
    # Test shuffle
    # board = start_board
    # print("Start board:", game.stringRepresentation(board))
    # print()
    # while game.getGameEnded(board, curr_player) == 0:
    #     # do random action from legal actions
    #     legal_actions = game.getValidMoves(board, curr_player)
    #     action = random.choice([i for i, x in enumerate(legal_actions) if x == 1 and i != 3])
    #     actions = ["raise", "call", "check", "fold"]

    #     board, curr_player = game.getNextState(board, curr_player, action)
    #     canonical_board = game.getCanonicalForm(board, curr_player)
    #     board, pi = game.getSymmetries(canonical_board, [.25] * 4)[0]
    #     # simulate a game where winning player wins 500
    
    # first_board = game.getCanonicalForm(board, 1)
    # print("First board:", game.stringRepresentation(first_board))
    # print()
    # time.sleep(1)

    # # Check same thing again, see if its different
    # board = start_board
    # curr_player = 1
    # print("Start board:", game.stringRepresentation(board))
    # print()
    # while game.getGameEnded(board, curr_player) == 0:
    #     # do random action from legal actions
    #     legal_actions = game.getValidMoves(board, curr_player)
    #     action = random.choice([i for i, x in enumerate(legal_actions) if x == 1 and i != 3])
    #     actions = ["raise", "call", "check", "fold"]

    #     board, curr_player = game.getNextState(board, curr_player, action)
    #     canonical_board = game.getCanonicalForm(board, curr_player)
    #     board, pi = game.getSymmetries(canonical_board, [.25] * 4)[0]
    #     # simulate a game where winning player wins 500
    
    # second_board = game.getCanonicalForm(board, 1)
    # print("Second board:", game.stringRepresentation(second_board))