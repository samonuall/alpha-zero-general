import logging
import concurrent.futures
from tqdm import tqdm

log = logging.getLogger(__name__)


#TODO: parallelize arena
class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                log.debug(f'board = {board}')
                log.debug(f"string representation = {self.game.stringRepresentation(self.game.getCanonicalForm(board, curPlayer))}")
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, num_workers=1, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0


        # for _ in tqdm(range(num), desc="Arena.playGames (1)"):
        #     gameResult = self.playGame(verbose=verbose)
        #     if gameResult == 1:
        #         oneWon += 1
        #     elif gameResult == -1:
        #         twoWon += 1
        #     else:
        #         draws += 1

        # self.player1, self.player2 = self.player2, self.player1

        # for _ in tqdm(range(num), desc="Arena.playGames (2)"):
        #     gameResult = self.playGame(verbose=verbose)
        #     if gameResult == -1:
        #         oneWon += 1
        #     elif gameResult == 1:
        #         twoWon += 1
        #     else:
        #         draws += 1

        
        log.info(f"Playing {num*2} games with {num_workers} workers...")
        if num_workers > 1:
            # First half: player1 starts
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_game = {
                    executor.submit(self.playGame, verbose): i 
                    for i in range(num)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_game), 
                                total=num, desc="Arena.playGames (1)"):
                    
                    gameResult = future.result()
                    if gameResult > .01:
                        oneWon += 1
                    elif gameResult < 0.01:
                        twoWon += 1
                    else:
                        draws += 1
        else:
            for _ in tqdm(range(num), desc="Arena.playGames (1)"):
                gameResult = self.playGame(verbose=verbose)
                if gameResult > 0.01:
                    oneWon += 1
                elif gameResult < 0.01:
                    twoWon += 1
                else:
                    draws += 1
        
        self.player1, self.player2 = self.player2, self.player1
        
        # Second half: player2 starts
        if num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_game = {
                    executor.submit(self.playGame, verbose): i 
                    for i in range(num)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_game), 
                                total=num, desc="Arena.playGames (2)"):
                    
                    gameResult = future.result()
                    if gameResult < 0.01:
                        oneWon += 1
                    elif gameResult > 0.01:
                        twoWon += 1
                    else:
                        draws += 1
        else:
            for _ in tqdm(range(num), desc="Arena.playGames (2)"):
                gameResult = self.playGame(verbose=verbose)
                if gameResult < 0.01:
                    oneWon += 1
                elif gameResult > 0.01:
                    twoWon += 1
                else:
                    draws += 1

        return oneWon, twoWon, draws
