import chess


# Initialize a chess board
board = chess.Board()


def print_move(notation):
    return str(chess.Move.from_uci(notation))