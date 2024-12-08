import chess

class myEngine():
    def __init__(self):
        self.board = chess.Board()
        self.last_board = None

    def move(self, move):
        self.last_board = self.board
        self.board.push(move)

    def engine_connect(self):
        pass