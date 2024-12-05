import dearpygui.dearpygui as dpg
import os

import mychess

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

column_style = "\ta\t\t  b\t\t  c\t\t  d\t\t  e\t\t f\t\t  g\t\t h"
column_format = [x.strip('\t ') for x in column_style.split('\t ')]

# dict to hold textures
chess_pieces = {}
piece_scale = 0.55

# Chess move notation example
chess_moves = []

# Algebraic notation
click_num = 0
current_move = ""

# Function to load an image as a texture and store it in the dictionary
def load_image_as_texture(image_path, tag):
    width, height, channels, image_data = dpg.load_image(image_path)
    
    # Add texture only if it doesn't already exist in the dictionary
    if tag not in chess_pieces:
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width, height, image_data, tag=tag)
        chess_pieces[tag] = (width, height)

# Function to draw the image from the dictionary if the key exists
def draw_image_from_texture(tag, x_pos, y_pos):
    if tag in chess_pieces:
        width, height = chess_pieces[tag]
        dpg.draw_image(tag, [x_pos, y_pos], [x_pos + int(width * piece_scale), y_pos + int(height * piece_scale)])

def valid_click(val):
    ofs = 10
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

        global click_num, current_move
        if(valid_click(relative_pos[0]) and valid_click(relative_pos[1])):
            current_move = current_move + column_format[int(relative_pos[0] / square_size) % 8] + str(8 - int(relative_pos[1] / square_size) % 8)
            click_num += 1

            #print(current_move)
            if(click_num == 2):
                chess_moves.append(current_move)
                current_move = ""
                click_num = 0
                update_chess_moves()
        elif(click_num == 1):
            click_num = 0
            current_move = ""
        



# Function to update the displayed moves in the scrollable text
def update_chess_moves():
    # Clear previous content in the child window and add updated moves
    with dpg.handler_registry():
        dpg.delete_item("moves_holder")  # Delete previous content (if any)
    
    with dpg.group(tag="moves_holder", parent= "notation"):
        for m in range(0, len(chess_moves) - 2, 2):
            dpg.add_text(str(m // 2 + 1) + ") " + chess_moves[m] + "\t" + chess_moves[m + 1] + "\n")
        
        if(len(chess_moves) % 2):
            dpg.add_text(str(len(chess_moves) // 2 + 1) + ") " + chess_moves[len(chess_moves) - 1] + "\t\n")
        else:
            dpg.add_text(str(len(chess_moves) // 2) + ") " + chess_moves[len(chess_moves) - 2] + "\t" + chess_moves[len(chess_moves) - 1] + "\n")

# Create a resizable window
def create_window():
    with dpg.window(label="board window", tag="board_Window", pos= (0, 0), no_resize= True,
                    no_collapse= True, no_close= True, no_move= True,
                    width = screenWidth * r1, height= screenWidth):
        with dpg.drawlist(width=board_width, height=board_height):
            for row in range(board_size):
                i = j = 0
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
                    

                    for p in chess_pieces:
                        draw_image_from_texture(p, square_size * i, square_size * j)
                        i += 1
                        if(i > 7):
                            i = 0
                            j += 1

        dpg.add_text(column_style.upper())

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

# Function to load individual pieces
def load_pieces():
    for row in range(p_slice.piece_grid_height):
        for col in range(p_slice.piece_grid_width):
            piece_name = p_slice.color_name[row] + p_slice.chess_name[col]  # For simplicity, name pieces based on their grid position
            load_image_as_texture(p_slice.directory + "\\" + piece_name + ".png" , piece_name)
    #print("pieces loaded")
load_pieces()

def main():
    # Create the window
    create_window()

    # Configure viewport to be resizable and set up a callback for resizing
    dpg.create_viewport(title="chess", width= screenWidth, height= screenHeight, 
                        max_width= screenWidth, max_height= screenHeight, resizable=False)

    # Setup and start the Dear PyGui rendering loop
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
main()

# Cleanup Dear PyGui context after use
dpg.destroy_context()