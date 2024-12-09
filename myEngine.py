import chess

class myEngine():
    def __init__(self):
        self.board = chess.Board()
        self.chess_moves = []
        self.white_move = True

    def reset_board(self):
        self.board.reset_board()
        self.chess_moves.clear()

    def move(self, move):
        self.board.push(move)
        self.white_move = not self.white_move

    def engine_connect(self):
        pass