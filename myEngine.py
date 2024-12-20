import chess
import random as rd

import bot_model as bm

class myEngine():
    def __init__(self, name =""):
        self.name = name
        self.board = chess.Board()
        self.chess_moves = []
        self.white_move = True

        self.loaded_data = False
        self.enabled = False

        self.game_result = None

        self.engine_white = True # if engine plays white 

    def check_game(self):
        if(self.board.is_fivefold_repetition() or self.board.is_insufficient_material() or self.board.is_stalemate()):
            self.game_result = "draw"

        if(self.board.is_checkmate()):
            if(self.board.turn == chess.WHITE):
                self.game_result = "Black won"
            elif(self.board.turn == chess.BLACK):
                self.game_result = "White won"       

    def reset_board(self):
        self.board.reset()
        self.chess_moves.clear()

        self.enabled = False
        self.game_result = None

        self.white_move = True

    def move(self, move: chess.Move):
        self.board.push(move)
        self.chess_moves.append(move.uci)

        self.white_move = not self.white_move
        self.check_game()

    def autoplay(self):
        self.engine_white = self.white_move
        self.enabled = not self.enabled

    def evaluate(self, timer):
        if(self.name == ""):
            return bm.simple_bot(self.board.fen())

    def best_move(self, timer = None):
        self.check_game()

        if(self.game_result != None):
            self.enabled = not self.enabled
            return

        if(self.engine_white and self.white_move):
            move = self.evaluate(timer)
            self.move(self.board.parse_san(move), move)
        elif not (self.engine_white or self.white_move):
            move = self.evaluate(timer)
            self.move(self.board.parse_san(move), move)
            
    def random_move(self):
        self.check_game()

        if(self.game_result != None):
            return

        move = rd.choice(list(self.board.generate_legal_moves()))
        self.move(move, move.uci())


# engine
#model = myEngine()  
