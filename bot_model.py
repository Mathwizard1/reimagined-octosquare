import torch
import torch.nn as nn

import numpy as np

import threading

import chess
from chess import Board

import heapq
import regex as re

evaluator_directory = "weights\\evaluator_"

# replace checkmates with huge evals
limit_val = 200

# same embeddings used for evaluator_train
def board2d_embed(game: Board):    
    pass

# citation: https://www.youtube.com/watch?v=aOwvRvTPQrs
def bitboard_embed(game: Board):
    pieces = ('k', 'q', 'r', 'b', 'n', 'p')
    layers = []

    for piece in pieces:
        b = str(game)
        b = re.sub(f'[^{piece}{piece.upper()} \n]','.', b)
        b = re.sub(f'{piece}', '-1', b) 
        b = re.sub(f'{piece.upper()}', '1', b)
        b = re.sub(f'\.', '0', b)

        board = []
        for row in b.split('\n'):
            row = row.split(' ')
            row = [int(x) for x in row]
            board.append(row)

        layers.append(np.array(board))

    board_embed = np.stack(layers)
    return torch.tensor(board_embed, dtype= torch.float32)

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
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2 (8x8 -> 4x4)
            
            # Second convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2 (4x4 -> 2x2)

            # Flatten the 2x2x64 output to a 1D vector (256)
            nn.Flatten(start_dim= 0),
            
            # Fully connected layer 1
            nn.Linear(256, 128),
            nn.LeakyReLU(),

            # Fully connected layer 2
            nn.Linear(128, 128),
            nn.ReLU(),

            # Fully connected layer 3
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer (single value for regression)
            nn.Linear(64, 1)
        )

        return model_list
    
    ### NEEDS WORK
    def intui_bot():
        model_list = (

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

        # Start the thread
        thread = threading.Thread(target= self.instanced_bot, kwargs={'self': self,'game': game, 'result': result}, daemon= True)
        thread.start()

        # Wait for the thread to finish or timeout
        thread.join(timer)
        print("thread over")

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
        pass

if(__name__ == '__main__'):
    bot_test = bots()

    game = Board()

    move = bot_test.active_Bot(game.board_fen())
    #game.push_san(move)

    #move = bot_test.active_Bot(game.board_fen())