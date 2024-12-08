import dearpygui.dearpygui as dpg
import os

import chess
from myEngine import myEngine

import piece_slicer as p_slice

if not os.path.exists(p_slice.directory):
    os.makedirs(p_slice.directory, exist_ok=True)  # exist_ok=True will not raise an error if the directory already exists
    p_slice.slice_pieces()

#screen setting
screenWidth, screenHeight = 1280, 720
r1 = 1 / 2
r2 = 1 / 3
r3 = 2 / 5

# Chessboard settings
board_size = 8  # 8x8 chessboard
square_size = 75  # Size of each square
board_width = board_size * square_size
board_height = board_size * square_size

column_style = " \ta \t\t b \t\t c \t\t d \t\t e \t\t f \t\t g \t\t h\t "
column_format = [x.strip('\t ') for x in column_style.split('\t ')]
flipped = 0

# dict to hold textures
chess_pieces = {}
piece_scale = 0.55

# Chess move notation example
chess_moves = []

# Algebraic notation
click_num = 0
current_move = ""

# small offset
ofs = 10

# engine
engine = myEngine()

# Function to load an image as a texture and store it in the dictionary
def load_image_as_texture(image_path, tag):
    width, height, channels, image_data = dpg.load_image(image_path)
    
    # Add texture only if it doesn't already exist in the dictionary
    if tag not in chess_pieces:
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width, height, image_data, tag=tag)
        chess_pieces[tag] = (width, height)

def valid_click(val):
    if(val > ofs and val < board_size * square_size + ofs):
        return True
    return False

# Callback function to get the mouse position relative to the window
def get_mouse_click_position(sender, app_data, user_data):
    # Check if the mouse is inside the window when clicked
    mouse_pos = dpg.get_mouse_pos()  # Get global mouse position
    window_pos = dpg.get_item_pos(user_data)  # Get window position
    window_size = dpg.get_item_rect_size(user_data)  # Get window size

    # Check if the click happened inside the window
    if (window_pos[0] <= mouse_pos[0] <= window_pos[0] + window_size[0] and 
       window_pos[1] <= mouse_pos[1] <= window_pos[1] + window_size[1]):
        
        # Calculate the relative mouse position inside the window
        relative_pos = (mouse_pos[0] - window_pos[0], mouse_pos[1] - window_pos[1])
        #print(f"Mouse clicked at: {relative_pos} (relative to the window)")

        valid_move = False
        #not_pawn = False

        global click_num, current_move
        if(valid_click(relative_pos[0]) and valid_click(relative_pos[1])):
            r = int(relative_pos[0] / square_size)
            f = int(relative_pos[1] / square_size)
            square_val = ((1 - flipped) * r + (flipped) * (7 - r)) + ((1 - flipped) * (7 - f) + (flipped) * f) * 8

            if(engine.board.piece_at(square_val)):
                valid_move = True
                '''p = engine.board.piece_at(square_val).symbol().upper()
                if(click_num == 0 and (p != "P" and p != "")):
                    not_pawn = True
                    current_move += p'''
            click_num += 1
            current_move += chess.square_name(square_val)

            color = (255, 0, 0)
            #print(current_move)

            if(click_num == 2):
                try:
                    input_move = engine.board.parse_san(current_move) #[1:] if(not_pawn) else current_move)

                    valid_move = True
                    engine.move(input_move)
                    chess_moves.append(current_move)

                    click_num = 0
                    current_move = ""
                    color = (0, 255, 0)

                    update_chess_moves()
                except:
                    click_num = 0
                    current_move = ""
                    valid_move = False

        elif(click_num == 1):
            click_num = 0
            current_move = ""

        dpg.delete_item("tempRect")
        if(valid_move):
            top_left_x = int(relative_pos[0] / square_size) * square_size
            top_left_y = int(relative_pos[1] / square_size) * square_size
            dpg.draw_rectangle(
                pmin=(top_left_x, top_left_y), 
                pmax=(top_left_x + square_size, top_left_y + square_size), 
                fill= color, tag= "tempRect", parent= "board_squares")

        draw_pieces()



# Function to update the displayed moves in the scrollable text
def update_chess_moves():
    # Clear previous content in the child window and add updated moves
    dpg.delete_item("moves_holder") 
    
    with dpg.group(tag="moves_holder", parent= "notation"):
        for m in range(0, len(chess_moves) - 2, 2):
            dpg.add_text(str(m // 2 + 1) + ") " + chess_moves[m] + "\t" + chess_moves[m + 1] + "\n")
        
        if(len(chess_moves) % 2):
            dpg.add_text(str(len(chess_moves) // 2 + 1) + ") " + chess_moves[len(chess_moves) - 1] + "\t\n")
        else:
            dpg.add_text(str(len(chess_moves) // 2) + ") " + chess_moves[len(chess_moves) - 2] + "\t" + chess_moves[len(chess_moves) - 1] + "\n")

def convert_symbol(symb: str):
    name = ""

    if(symb.islower()):
        name = p_slice.color_name[0]
    else:
        symb = symb.lower()
        name = p_slice.color_name[1]

    symb_map = {
        "k": "King",
        "q": "Queen",
        "r": "Rook",
        "b": "Bishop",
        "n": "Knight",
        "p": "Pawn"
    }

    return name+symb_map[symb]

# Callback function for the button
def board_flipper(sender, app_data):
    dpg.delete_item("file_text")
    dpg.delete_item("rank_text")

    global flipped
    if(flipped == 1):
        dpg.add_text(column_style.upper(), tag= "file_text", parent= "board_display")
        with dpg.group(tag= "rank_text", parent= "board_display"):
            for i in range(9, 0, -1):
                dpg.add_text(str(i), pos=(board_width + ofs, square_size * (9 - i - 0.3)))
        flipped = 0
    else:
        dpg.add_text(column_style[::-1].upper(), tag= "file_text", parent= "board_display")
        with dpg.group(tag= "rank_text", parent= "board_display"):
            for i in range(9, 0, -1):
                dpg.add_text(str(9 - i), pos=(board_width + ofs, square_size * (9 - i - 0.3)))
        flipped = 1

    draw_pieces()

# draw position
def draw_pieces():
    # erase
    dpg.delete_item("piece_position")

    # draw
    with dpg.draw_layer(tag= "piece_position", parent= "board_squares"):
        # Iterate over all squares (0 to 63)
        for square in chess.SQUARES:
            piece = engine.board.piece_at(square)

            if piece:
                # Get the rank and file of the square
                rank = (1 - flipped) * (7 - chess.square_rank(square)) + (flipped) * chess.square_rank(square)
                file = (1 - flipped) * chess.square_file(square) + (flipped) * (7 - chess.square_file(square))

                tag = convert_symbol(piece.symbol())

                if tag in chess_pieces:
                        width, height = chess_pieces[tag]
                        dpg.draw_image(tag, [square_size * file, square_size * rank], [square_size * file + int(width * piece_scale), square_size * rank + int(height * piece_scale)], parent= "piece_position")


# Create a resizable window
def create_window():
    with dpg.window(label="board window", tag="board_Window", pos= (0, 0), no_resize= True,
                    no_collapse= True, no_close= True, no_move= True,
                    width = screenWidth * r1, height= screenWidth):
        
        with dpg.group(tag= "board_display"):
            with dpg.drawlist(width=board_width, height=board_height, tag= "board_squares"):
                for row in range(board_size):
                    for col in range(board_size):
                        # Determine color based on row and column index (alternate between white and black)
                        color = p_slice.white_color if (row + col) % 2 == 0 else p_slice.brown_color
                        # Calculate the square position
                        top_left_x = col * square_size
                        top_left_y = row * square_size
                        bottom_right_x = top_left_x + square_size
                        bottom_right_y = top_left_y + square_size
                        # Draw the square
                        dpg.draw_rectangle(
                            pmin=(top_left_x, top_left_y), 
                            pmax=(bottom_right_x, bottom_right_y), 
                            color=(0, 0, 0, 255),  # Optional border color (black)
                            fill=color)
                draw_pieces()

            # Add column with row numbers (1 to 8)
            with dpg.group(tag= "rank_text", parent= "board_display"):
                for i in range(9, 0, -1):
                    dpg.add_text(str(i), pos=(board_width + ofs, square_size * (9 - i - 0.3)))  # Position each number in a column
        
            # Add file in a row
            dpg.add_text(column_style.upper(), tag= "file_text", parent= "board_display")

        with dpg.group(tag = "board_buttons", horizontal= True):
            dpg.add_button(label="Flip Board", callback= board_flipper)
            dpg.add_button(label="New Board")

    with dpg.window(label="notation window", tag="notation_Window", pos= (int(screenWidth * r1), 0), no_resize= True,
                    no_collapse= True, no_close= True, no_move= True,
                    width = int(screenWidth * (1 - r1) * r2), height= int(screenHeight * r3)):
        # Create a scrollable child window to display the moves
        with dpg.child_window(width = int(screenWidth * (1 - r1) * (r2 - 0.03)), border=True, tag= "notation"):
            # Display the formatted chess moves
            dpg.add_text("  WHITE\tBLACK\n")
            dpg.add_text("No moves\tNo moves", tag = "moves_holder", parent= "notation")

    # Set a handler for mouse clicks
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=get_mouse_click_position, user_data="board_Window")

    with dpg.window(label="engine window", tag="engine_Window", pos= (int(screenWidth * (r1 + r2 - r1 * r2)), 0), no_resize= True,
                        no_collapse= True, no_close= True, no_move= True,
                        width = int(screenWidth * (1 - r1) * (1 - r2)), height= int(screenHeight * r3)):
        pass

    with dpg.window(label="evaluation window", tag="evaluation_Window", pos= (int(screenWidth * r1), int(screenHeight * r3)), no_resize= True,
                    no_collapse= True, no_close= True, no_move= True,
                    width = int(screenWidth * (1 - r1)), height= int(screenWidth * (1 - r3))):
        pass

# Initialize Dear PyGui
dpg.create_context()

#load individual pieces
for row in range(p_slice.piece_grid_height):
    for col in range(p_slice.piece_grid_width):
        piece_name = p_slice.color_name[row] + p_slice.chess_name[col]  # For simplicity, name pieces
        load_image_as_texture(p_slice.directory + "\\" + piece_name + ".png" , piece_name)


# Create the window
create_window()

# Configure viewport to be resizable and set up a callback for resizing
dpg.create_viewport(title="chess", width= screenWidth, height= screenHeight, 
                    max_width= screenWidth, max_height= screenHeight, resizable=False)

# Setup and start the Dear PyGui rendering loop
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

# Cleanup Dear PyGui context after use
dpg.destroy_context()