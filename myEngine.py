import chess
import random as rd

import bot_model as bm

class myEngine():
    def __init__(self, name ="simple_bot"):
        self.bot = bm.bots(name)

        self.board = chess.Board()
        self.chess_moves = []
        self.white_move = True

        self.enabled = False

        self.game_result = None

        self.engine_white = True # if engine plays white 

    def check_game(self):
        if(self.board.is_fivefold_repetition() or self.board.is_insufficient_material() or self.board.is_stalemate()):
            self.game_result = "draw (0.5 : 0.5)"
            self.enabled = False

        if(self.board.is_checkmate()):
            if(self.board.turn == chess.WHITE):
                self.game_result = "Black won (0 : 1)"
            elif(self.board.turn == chess.BLACK):
                self.game_result = "White won (1 : 0)"
            self.enabled = False      

    def reset_board(self):
        self.board.reset()
        self.chess_moves.clear()

        self.enabled = False
        self.game_result = None

        self.white_move = True

    def move(self, move: chess.Move):
        self.board.push(move)
        self.chess_moves.append(move.uci())

        self.white_move = not self.white_move
        self.check_game()

    def setup_board(self, fen):
        self.enabled = False
        board = chess.Board()

        try:
            board.set_fen(fen)
            
            self.white_move = True if(board.turn == chess.WHITE) else False
            self.board = board
            self.chess_moves.clear()
        except:
            pass

        self.check_game()

    def autoplay(self):
        self.engine_white = self.white_move
        self.enabled = not self.enabled

    def get_move(self, timer):
        return self.bot.active_Bot(self.board.fen(), timer)

    def best_move(self, timer = 0):
        self.check_game()

        if(self.game_result != None):
            return

        if(self.engine_white and self.white_move):
            move = self.get_move(timer)
            self.move(self.board.parse_san(move))
        elif not (self.engine_white or self.white_move):
            move = self.get_move(timer)
            self.move(self.board.parse_san(move))
            
    def random_move(self):
        self.check_game()

        if(self.game_result != None):
            return

        move = rd.choice(list(self.board.generate_legal_moves()))
        self.move(move)


# engine
#model = myEngine()  
