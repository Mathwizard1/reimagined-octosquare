import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from chess import Board
import pandas as pd

import os.path as path
from bot_model import evaluator, bitboard_embed

# loading chess data
filtered_df = pd.read_csv('data\\random_evals.csv', nrows= 50000)

# Pre processing

# All checkmate position
#mask = df['Evaluation'].str.contains('#')
#filtered_df = df[mask == True]

# Normal position
#filtered_df = df[mask == False]

# All position
limit_val = 20000

filtered_df.loc[filtered_df['Evaluation'].str.contains('#-'), 'Evaluation'] = str(-1 * limit_val)  
filtered_df.loc[filtered_df['Evaluation'].str.contains('#+'), 'Evaluation'] = str(limit_val)

filtered_df['Evaluation'] = pd.to_numeric(filtered_df['Evaluation'])

print("number of samples:", len(filtered_df))

board_data, targets = [], []

#### Statistical scaling #####

'''def Robust_Scaling():
    median = filtered_df['Evaluation'].median()
    iqr = filtered_df['Evaluation'].quantile(0.75) - filtered_df['Evaluation'].quantile(0.25)

    filtered_df['scaled'] = (filtered_df['Evaluation'] - median) / iqr

def z_score():
    filtered_df['Evaluation'] = (filtered_df['Evaluation'] - filtered_df['Evaluation'].mean()) / filtered_df['Evaluation'].std()
'''
##############################

#z_score()

def game_board(fen):
    game = Board(fen)
    return bitboard_embed(game)

for row in filtered_df.itertuples():
    board_data.append(game_board(row.FEN))
    targets.append(row.Evaluation)

    #print(board_data[0], len(board_data[0]))
    #print(targets[0])
    #exit()


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


##################################################
##################################################

###### model parameters ######

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

current_model = "simple_bot"

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

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
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
        model.load_state_dict(torch.load(model.name, weights_only= True))
        print(f"Model loaded for epoch {epoch+1}")

# Train the model
train_model(model, dataloader, num_epochs=5)