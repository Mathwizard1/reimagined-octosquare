import torch
import torch.nn as nn

import numpy as np

import chess
from chess import Board

import regex as re

evaluator_directory = "weights\\evaluator_"

# replace checkmates with huge evals
limit_val = 200

# same embeddings used for evaluator_train
def board2d_embed(game: Board):    
    symb_map = {
        'k': 10,
        'q': 9,
        'r': 5,
        'b': 3.1,
        'n': 2.9,
        'p': 1
    }

    Board = []
    for square in chess.SquareSet:
        piece = game.piece_at(square)

        if(piece != None):
            symb = piece.symbol()
            val= 1 if(symb.isupper()) else -1
            Board.append(symb_map[symb.lower()] * val)
        else:
            Board.append(0)
    board_embed = np.array(Board).reshape(8,8)
    return torch.tensor(board_embed, dtype= torch.float32)

# citation: https://www.youtube.com/watch?v=aOwvRvTPQrs
def bitboard_embed(game: Board):
    pieces = ('k', 'q', 'r', 'b', 'n', 'p')
    layers = []

    for piece in pieces:
        b = str(game)
        b = re.sub(f'[^{piece}{piece.upper()} \n]','.', b)
        b = re.sub(f'{piece}', '-1', b) 
        b = re.sub(f'{piece.upper()}', '1', b)
        b = re.sub(r'\.', '0', b)

        board = []
        for row in b.split('\n'):
            row = row.split(' ')
            row = [int(x) for x in row]
            board.append(row)

        layers.append(np.array(board))

    board_embed = np.stack(layers)
    return torch.tensor(board_embed, dtype= torch.float32)

# Special attention block to be used inside the sequential
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# evaluator class for positions
class evaluator(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = evaluator_directory + name + ".pth"

        print(name,"model")
        temp_model = getattr(evaluator, name)()
        self.instanced_model = nn.Sequential(*temp_model)

    def forward(self, x):
        '''for layer in self.isinstance_model:
            x = layer(x)

            print(x.shape)
        return x'''

        return self.instanced_model(x)

    def simple_bot():
        model_list = (
            # First convolutional layer
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2 (8x8 -> 4x4)
            
            # Second convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2 (4x4 -> 2x2)

            # Flatten the 2x2x64 output to a 1D vector (256)
            nn.Flatten(start_dim= 0),
            
            # Fully connected layer 1
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope= 0.1),

            # Fully connected layer 2
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope= 0.1),

            # Fully connected layer 3
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer (single value for regression)
            nn.Linear(64, 1)
        )

        return model_list
    
    ### NEEDS WORK
    def intui_bot():
        embed_dim = 64
        num_heads= 4

        model_list = (
            nn.Linear(8 * 8, embed_dim),  # Embedding layer for input (8x8)
            AttentionBlock(embed_dim=embed_dim, num_heads=num_heads),  # Attention layer as a block

            nn.Flatten(start_dim= 1),  # Flatten attention output

            nn.Linear(embed_dim * 8 * 8, 128),  # Fully connected layers
            nn.LeakyReLU(negative_slope= 0.1),

            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope= 0.1),

            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope= 0.1),

            nn.Linear(64, 1)  # Output layer for regression
        )

        return model_list

class bots():
    def __init__(self, name = ""):
        if(name in dir(bots) and "_bot" in name):
            self.instanced_bot = getattr(bots, name)
        else:
            self.instanced_bot = getattr(bots, "simple_bot")

        self.evaluator = None
        self.load_evaluator()

    def load_evaluator(self):
        ## choose the right model for the bot to work
        self.evaluator = evaluator(self.instanced_bot.__name__)
        self.evaluator.load_state_dict(
            torch.load(self.evaluator.name, weights_only= True))
        
        # Set evaluation mode
        self.evaluator.eval()
        
    #def move_Reorder(self, game: Board) -> list:
    #    pass

    def active_Bot(self, fen, timer = 10) -> chess.Move.uci:
        game = Board(fen)
        result = {'final_move' : None}

        self.instanced_bot(game, result)

        return result['final_move']

    def simple_bot(self, game: Board, result = {}):
        moves = list(game.generate_legal_moves())
        move_reserve = []

        white_move = True if(game.turn == chess.WHITE) else False
        move_eval = -np.inf if(game.turn == chess.WHITE) else np.inf

        # checks, captures and promotion
        for move in moves:
            if(game.gives_check(move) or move.promotion != None):
                game.push(move)
                game_eval = self.evaluator(bitboard_embed(game))
                print(game_eval, move.uci())
                game.pop()

                if(white_move and game_eval > move_eval):
                    move_eval = game_eval
                    result['final_move'] = move.uci()
                elif(not white_move and game_eval < move_eval):
                    move_eval = game_eval
                    result['final_move'] = move.uci()
            else:
                move_reserve.append(move)

        # other moves
        for move in move_reserve:
            game.push(move)
            game_eval = self.evaluator(bitboard_embed(game))
            print(game_eval, move.uci())
            game.pop()

            if(white_move and game_eval > move_eval):
                move_eval = game_eval
                result['final_move'] = move.uci()
            elif(not white_move and game_eval < move_eval):
                move_eval = game_eval
                result['final_move'] = move.uci()


    ### NEEDS WORK
    def intui_bot(self, game: Board, result = {}):
        moves = list(game.generate_legal_moves())
        move_reserve = []

        white_move = True if(game.turn == chess.WHITE) else False
        move_eval = -np.inf if(game.turn == chess.WHITE) else np.inf

        # checks, captures and promotion
        for move in moves:
            if(game.gives_check(move) or move.promotion != None):
                game.push(move)
                game_eval = self.evaluator(bitboard_embed(game))
                print(game_eval, move.uci())
                game.pop()

                if(white_move and game_eval > move_eval):
                    move_eval = game_eval
                    result['final_move'] = move.uci()
                elif(not white_move and game_eval < move_eval):
                    move_eval = game_eval
                    result['final_move'] = move.uci()
            else:
                move_reserve.append(move)

        # other moves
        for move in move_reserve:
            game.push(move)
            game_eval = self.evaluator(bitboard_embed(game))
            print(game_eval, move.uci())
            game.pop()

            if(white_move and game_eval > move_eval):
                move_eval = game_eval
                result['final_move'] = move.uci()
            elif(not white_move and game_eval < move_eval):
                move_eval = game_eval
                result['final_move'] = move.uci()

if(__name__ == '__main__'):
    bot_test = bots()

    game = Board()

    move = bot_test.active_Bot(game.board_fen())
    #game.push_san(move)

    #move = bot_test.active_Bot(game.board_fen())