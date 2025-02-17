
# Chess Bot Development Environment

#### By Anshurup gupta
This project is a development environment for creating, testing, and comparing simple chess bots built with different neural architectures using PyTorch, Python-Chess, and DearPyGui. The environment is designed to facilitate easy experimentation with various neural network models and to evaluate their performance in chess games.

### Key Features
- *Modular Design*: Easily integrate various neural architectures (e.g., CNNs, MLPs) with PyTorch for training chess bots.
- *Python-Chess Integration*: Utilize the Python-Chess library to handle the game logic, move validation, and game state management.
- *Graphical User Interface*: Use DearPyGui to create a user-friendly interface for configuring the bot's parameters and running simulations.
- *Training and Evaluation*: Train the bots using PyTorch with custom datasets, or load pre-trained models. Evaluate performance against human players or other bots.

### Libraries
- PyTorch (for creating and training neural network models)
- Python-Chess (for chess game logic)
- DearPyGui (for the graphical interface)

### Installation
- Clone the repository:
git clone https://github.com/Mathwizard1/reimagined-octosquare.git

- Install dependencies:
pip install -r requirements.txt

- Run the application:
python gui.py

### Running Simulations
- *Configure Bots*: Launch the application using python main.py. Use the DearPyGui interface to select or configure different bots, neural architectures, and game parameters.

### Future Features
Better trained weights
Test bots against each other
Create Elo system
Additional neural architectures like Transformers for better decision-making.
More sophisticated evaluation metrics.
Support for human players to challenge AI bots via the GUI.

### Contributing
If you wish to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request.
