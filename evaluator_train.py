import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from chess import Board
import pandas as pd

import os.path as path
from bot_model import evaluator, bitboard_embed, board2d_embed, limit_val

#############################################################

# loading chess data
filtered_df = pd.read_csv('data\\random_evals.csv', nrows= 10000)

#############################################################

# Pre processing
#print(filtered_df.dtypes)

# All checkmate position
#mask = filtered_df['Evaluation'].astype(str).str.contains('#', na= False)
#filtered_df = filtered_df[mask == True]

# Normal position
#filtered_df = filtered_df[mask == False]

# All position
#if(mask.any()):
    #print("Checkmate positions found")
filtered_df.loc[filtered_df['Evaluation'].str.contains('#-'), 'Evaluation'] = str(-1 * limit_val * 100)  
filtered_df.loc[filtered_df['Evaluation'].str.contains('#+'), 'Evaluation'] = str(limit_val * 100)

filtered_df['Evaluation'] = pd.to_numeric(filtered_df['Evaluation'])

#############################################################

# Scaling
def min_max_Scaling(f = 10):
    # 0 -> 10
    filtered_df['Evaluation'] = (
        filtered_df['Evaluation'] - filtered_df['Evaluation'].min()) / (
        filtered_df['Evaluation'].max() - filtered_df['Evaluation'].min()
        ) * f

min_max_Scaling(limit_val * 2)
#############################################################

board_data, targets = [], []

def game_board(fen, bit_embed = True):
    game = Board(fen)
    if(bit_embed):
        return bitboard_embed(game)
    else:
        return board2d_embed(game)

for row in filtered_df.itertuples():
    board_data.append(game_board(row.FEN))
    targets.append(row.Evaluation)

    #print(board_data[0], len(board_data[0]))
    #print(targets[0])
    #exit()

print("number of samples:", len(filtered_df))
#############################################################

# Create the dataset using the manually imported lists
class ManualDataset(Dataset):
    def __init__(self, features_list, targets_list):
        """
        Initialize dataset and split into train and test sets.
        
        :param features_list: List of feature rows (list of lists)
        :param targets_list: List of target values
        """

        # features and targets for training
        self.train_features = features_list
        self.train_targets = targets_list
    
    def __len__(self):
        # Return the size of the training data
        return len(self.train_targets)

    def __getitem__(self, idx):
        # Convert Python list to torch tensor for the training set
        features = self.train_features[idx]
        target = torch.tensor(self.train_targets[idx], dtype=torch.float32)
        return features, target

dataset = ManualDataset(board_data, targets)

# Create a DataLoader for batching
dataloader = DataLoader(dataset)

#############################################################
#############################################################

# model parameters

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

current_model = "intui_bot"

#############################################################

# Instantiate the model
model = evaluator(current_model)

# for loading pre-existing model
def load_pre_train():
    global model
    model.load_state_dict(torch.load(model.name, weights_only= True))

if(path.exists(model.name)):
    print(model.name,"loaded")
    load_pre_train()

#for param in model.parameters():
#    print(type(param), param.size())

#############################################################

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1.0e-6)

#############################################################

# Training

# Training function
def train_model(model : evaluator, dataloader, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        for batch_data, batch_target in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_data)
            loss = criterion(output, batch_target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print loss at the end of epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

        # Save model at the end of each epoch
        torch.save(model.state_dict(), model.name)
        print(f"Model saved after epoch {epoch+1}")

        # Load the model after saving to continue training in the next epoch
        if(epoch < num_epochs - 1):
            model.load_state_dict(torch.load(model.name, weights_only= True))
            print(f"Model loaded for epoch {epoch+2}")

# Train the model
train_model(model, dataloader, num_epochs= 2)