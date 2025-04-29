import sys
import os

# Add the parent directory of both alpha-poker and pypokerengine to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pypokerengine.api.emulator import Emulator
from testplayer import TestPlayer
from pypokerengine.engine.message_builder import MessageBuilder
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from alphaplayer import AlphaTrainPlayer
from poker.PokerLogic import PokerState, PokerBoard
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.engine.card import Card
from naiveplayer import NaivePlayer


def init_emulator(players_info, player_class):
    emulator = Emulator()
    
    emulator.set_game_rule(
        player_num = 2,
        max_round = 3,
        small_blind_amount = 5,
        ante_amount = 0,
    )

    uuids = list(players_info.keys())
    
    # register first player
    player = player_class()
    player.uuid = uuids[0]
    emulator.register_player(
        uuid = player.uuid,
        player = player
    )

    # register second player
    player = player_class()
    player.uuid = uuids[1]
    emulator.register_player(
        uuid = player.uuid,
        player = player
    )

    return emulator

def start_round(emulator, players_info):
    """Should return a pokerboard object with each player's state set"""
    board = PokerBoard(players_info)
    
    # Get initial state
    initial_game_state = emulator.generate_initial_game_state(players_info)

    # Start round
    print("Starting new round")
    game_state, events = emulator.start_new_round(initial_game_state)
    print("NEW ROUND", game_state["table"].seats.players[0].hole_card)
    print("NEW ROUND", game_state["table"].seats.players[1].hole_card)
    board.emulator_state = game_state
    
    uuids = list(players_info.keys())
    # Send round start message to all players
    msg = None
    uuids = list(players_info.keys())
    for i, player in enumerate(emulator.players_holder.values()):
        msg = MessageBuilder.build_round_start_message(game_state["round_count"], i, game_state["table"].seats)["message"]
        player.receive_round_start_message(game_state["round_count"], 
                                       msg["hole_card"], 
                                       [{"uuid": uuids[0]}, {"uuid": uuids[1]}])
        
        # Update poker states in board
        curr_state = board.player_states[player.uuid]
        for event in events:
            if event["type"] == "event_new_street":
                player.receive_street_start_message(event["street"], event["round_state"])
        
                curr_state["EHS"] = estimate_hole_card_win_rate(
                    30, 
                    2, 
                    [Card.from_str(card) for card in msg["hole_card"]], 
                    event["round_state"]["community_card"]
                )
                curr_state.set_hole(msg["hole_card"])
                curr_state["round_count"] = 1.0
                curr_state.set_pot(event["round_state"]["pot"]["main"]["amount"])
                curr_state.set_street(0)
                curr_state.set_community(event["round_state"]["community_card"])

        
         

    return board


if __name__ == "__main__":
    
    players_info = {
        "player1": {"stack": 1000, "name": "Alice"},
        "player2": {"stack": 1000, "name": "Bob"}
    }
    emulator = init_emulator(players_info, NaivePlayer)
    board = start_round(emulator, players_info)
    game_state = board.emulator_state
    
    round_done = False
    while not round_done:
        
        # Make player do the next action
        next_player_pos = game_state["next_player"]
        next_player_uuid = game_state["table"].seats.players[next_player_pos].uuid
        next_player_algorithm = emulator.fetch_player(next_player_uuid)
        msg = MessageBuilder.build_ask_message(next_player_pos, game_state)["message"]
        action = next_player_algorithm.declare_action(\
                msg["valid_actions"], msg["hole_card"], msg["round_state"])
        game_state, events = emulator.apply_action(game_state, action)
        

        print("HOLE CARD:", game_state["table"].seats.players[0].hole_card)
        print("GAME STATE", game_state)
       
       # Send messages for events
        for event in events:
            if event["type"] == "event_new_street":
                for player in emulator.players_holder.values():
                    player.receive_street_start_message(event["street"], event["round_state"])
            
            elif event["type"] == "event_round_finish":
                for player in emulator.players_holder.values():
                    player.receive_round_result_message(event["winners"], None, event["round_state"])
                    round_done = True


        