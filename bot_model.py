import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import threading
import csv

from Chessnut import Game

weight_directory = "weights\\"

class evaluator():
    def __init__(self, weights_path):
        

        pass

    def game_evaluation(self, game):


        pass

class bots():
    def __init__(self, name = "", weights= ""):

        if(name in dir(bots) and "_bot" in name):
            self.instanced_bot = getattr(bots, name)
        else:
            self.instanced_bot = getattr(bots, "simple_bot")

    def active_Bot(self, fen, timer):
        game = Game(fen)
        result = None

        #self.instanced_bot(game)

        return result

    def simple_bot(self, game, eval = 0):
        pass

    def intui_bot(self, game, eval = 0):
        pass

#bots()