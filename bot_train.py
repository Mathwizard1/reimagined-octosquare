import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import csv
from Chessnut import Game


directory = "weights\\"

###### Hyper parameters ######

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# number of samples from eval data
counter = 100

# batch size (< samples)
batch_size = 16

###### ###### ###### ######

def convert(symb: str):
    val = 1

    if(symb.islower()):
        val = -1
    else:
        symb = symb.lower()

    symb_map = {
        ' ': 0,
        'p': 1,
        'r': 5,
        'n': 2.9,
        'b': 3.1,
        'q': 9,
        'k': 50
    }

    return val * symb_map[symb]

def board_embed(fen_string):
    game = Game(fen=fen_string)  # Initialize the game with the FEN string
    
    board = []
    
    # Loop through each square (from a1 to h8)
    for rank in range(8):
        for file in range(8):
            square = file + (rank) * 8
            piece = game.board.get_piece(square)

            #print(piece)

            board.append(convert(piece))

        #board.append(row)
    
    #print(board)
    return board


# loading eval data
board_data, targets = [], []

with open("data\\random_evals.csv") as fp:
    csv_reader = csv.reader(fp)
    head_flag = True
    num = 0

    for row in csv_reader:
        if(head_flag):
            head_flag = False
            continue

        #print(row[0])
        #print(row[1])

        targets.append(int(row[1]))
        board_data.append(board_embed(row[0]))
        num += 1

        if(counter > 0 and num >= counter):
            break

    fp.close()

#print(board_data[0], len(board_data[0]))
#print(targets[0])

# Pre processing
class ManualDataset(Dataset):
    def __init__(self, features_list, targets_list, t_split=0.8):
        """
        Initialize dataset and split into train and test sets.
        
        :param features_list: List of feature rows (list of lists)
        :param targets_list: List of target values
        :param t_split: Proportion of data to use for training (default: 0.8)
        """
        self.features = features_list
        self.targets = targets_list
        
        # Calculate the index for splitting the data
        split_idx = int(len(features_list) * t_split)
        
        # Split features and targets into training and testing sets
        self.train_features = self.features[:split_idx]
        self.train_targets = self.targets[:split_idx]
        self.test_features = self.features[split_idx:]
        self.test_targets = self.targets[split_idx:]
    
    def __len__(self):
        # Return the size of the training data
        return len(self.train_targets)

    def __getitem__(self, idx):
        # Convert Python list to torch tensor for the training set
        features = torch.tensor(self.train_features[idx], dtype=torch.float32)
        target = torch.tensor(self.train_targets[idx], dtype=torch.float32)
        return features, target
    
    def get_test_data(self):
        """
        Return test data for validation.
        """
        test_features = torch.tensor(self.test_features, dtype=torch.float32)
        test_targets = torch.tensor(self.test_targets, dtype=torch.float32)
        return test_features, test_targets

# Create the dataset using the manually imported lists
dataset = ManualDataset(board_data, targets)

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=batch_size)

class evaluator(nn.Module):
    def __init__(self, name = ""):
        super().__init__()
        self.name = directory + "evaluator" + name + ".pth"
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


###### model parameters ######

# Instantiate the model
model = evaluator()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
###### ###### ###### ######

for epoch in range(epochs):
    for batch_features, batch_targets in dataloader:

        optimizer.zero_grad()
        predictions = model(batch_features).squeeze()
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % (epochs / 10) == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Validation step (after training)
model.eval()  # Set model to evaluation mode
test_features, test_targets = dataset.get_test_data()
with torch.no_grad():
    predictions = model(test_features).squeeze()
    for i in range(5):
        print(predictions[i], test_targets[i])
    val_loss = criterion(predictions, test_targets)
    print(f'Validation Loss: {val_loss.item():.4f}')

# Save the model
model_path = model.name
torch.save(model.state_dict(), model_path)
print("Model saved!")

# Load the model state dictionary
model = evaluator()
model.load_state_dict(torch.load(model_path, weights_only= True))

# Set the model to evaluation mode before inference
model.eval()

# Use the model for inference or validation
test_features, test_targets = dataset.get_test_data()
with torch.no_grad():
    predictions = model(test_features).squeeze()
    val_loss = criterion(predictions, test_targets)
    print(f'Validation Loss: {val_loss.item():.4f}')