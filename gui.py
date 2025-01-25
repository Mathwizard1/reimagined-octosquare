import dearpygui.dearpygui as dpg
import os

import chess
from myengine import myEngine

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
flipped = 0

# dict to hold textures
chess_pieces = {}
piece_scale = 0.55


# Algebraic notation
click_num = 0
current_notation = ""
hold_piece = ""

# small offset
ofs = 10

# engines
engine = myEngine()

def bot_selection(sender):
    print(f"{sender} bot selection Needs work")

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
def mouse_clicked(sender, app_data, user_data):
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
        
        r0 = int(relative_pos[0] / square_size)
        f0 = int(relative_pos[1] / square_size)

        window_actv = dpg.get_active_window()
        if(window_actv == dpg.get_alias_id("board_Window")):
            global click_num, current_notation, hold_piece
            color = (255, 0, 0)

            if(valid_click(relative_pos[0]) and valid_click(relative_pos[1])):
                r = (1 - flipped) * r0 + (flipped) * (7 - r0)
                f = (1 - flipped) * (7 - f0) + (flipped) * f0

                square_val = (r) + (f) * 8
                #print(f"[{r,f}]")

                if(click_num == 0):
                    if(engine.board.piece_at(square_val)):
                        hold_piece = engine.board.piece_at(square_val).symbol()
                    else:
                        hold_piece = ""
                        current_notation = ""
                        return
                
                if(click_num < 2):
                    click_num += 1
                    current_notation += chess.square_name(square_val)

                if(click_num > 1):
                    promotion_list = ["q","r","b","n"]
                    promotion_list = promotion_list if(f == 0) else [x.upper() for x in promotion_list]

                    if(click_num == 2 and 
                       ((hold_piece == "p" and f == 0) or (hold_piece == "P" and f == 7))):
                        click_num += 1
                        color = (0, 0, 255)

                        with dpg.draw_layer(tag= "promotion_pieces", parent= "board_squares"):
                            for x in range(2):
                                for y in range(2):
                                    top_left_x = (3 + y) * square_size
                                    top_left_y = (3 + x) * square_size
                                    dpg.draw_rectangle(
                                        pmin=(top_left_x, top_left_y), 
                                        pmax=(top_left_x + square_size, top_left_y + square_size), 
                                        fill= color, parent= "promotion_pieces")
                                    
                                    tag = convert_symbol(promotion_list[2 * x + y])
                                    width, height = chess_pieces[tag]
                                    dpg.draw_image(tag, [top_left_x, top_left_y], [top_left_x + int(width * piece_scale), top_left_y + int(height * piece_scale)], parent= "promotion_pieces")
                        
                    elif(click_num == 3):
                        if(3 <= r <= 4 and 3 <= f <= 4):
                            click_num -= 1
                            piece_chose = (1 - flipped) * ((r - 3) + 2 * (4 - f)) + flipped * ((4 - r) + 2 * (f - 3))
                            current_notation += '=' + promotion_list[piece_chose].upper()
                        else:
                            click_num = 0
                            current_notation = ""
                            hold_piece = ""

                            dpg.delete_item("tempRect")
                            dpg.delete_item("promotion_pieces")
                            return

                    #print(current_notation)
                    #print(click_num)

                    try:
                        input_move = engine.board.parse_san(current_notation)
                        
                        #print("valid")

                        engine.move(input_move)

                        color = (0, 255, 0)

                        update_chess_moves()
                    except:
                        #print("invalid")
                        if(click_num == 3):
                            return
                        hold_piece = ""

                    dpg.delete_item("promotion_pieces")

                    click_num = 0
                    current_notation = ""

            elif(click_num == 1):
                click_num = 0
                current_notation = ""
                hold_piece = ""

            dpg.delete_item("tempRect")
            if(hold_piece != ""):
                top_left_x = r0 * square_size
                top_left_y = f0 * square_size
                dpg.draw_rectangle(
                    pmin=(top_left_x, top_left_y), 
                    pmax=(top_left_x + square_size, top_left_y + square_size), 
                    fill= color, tag= "tempRect", parent= "board_squares")

            draw_pieces()

def submit_Fen(sender, app_data, user_data):
    user_input = dpg.get_value("string_input")
    #print(f"Submitted Text: {user_input}")

    engine.setup_board(user_input)
    update_chess_moves()
    draw_pieces()

# reset board to start
def board_start():
    global click_num, current_notation, hold_piece
    click_num = 0
    hold_piece = ""
    current_notation = ""
    
    dpg.delete_item("tempRect")
    dpg.delete_item("promotion_pieces")

    engine.reset_board()
    update_chess_moves()
    draw_pieces()

def engine_random():
    engine.random_move()
    update_chess_moves()
    draw_pieces()

def engine_auto():
    engine.autoplay()
    update_chess_moves()

def engine_automove():
    if(engine.enabled):
        user_input = dpg.get_value("eng_timer1")
        #print(f"timer: {user_input}")

        engine.best_move(user_input)
        update_chess_moves()
        draw_pieces()

# Function to update the displayed moves in the scrollable text
def update_chess_moves():
    # Clear previous content in the child window and add updated moves
    dpg.delete_item("moves_holder") 
    dpg.delete_item("to_move_holder")
    dpg.delete_item("autoplay")
    dpg.delete_item("over")

    moves_number = len(engine.chess_moves)
    with dpg.group(tag="moves_holder", parent= "notation"):
        for m in range(0, moves_number - 2, 2):
            dpg.add_text(str(m // 2 + 1) + ") " + engine.chess_moves[m] + "\t" + engine.chess_moves[m + 1] + "\n")
        
        if(moves_number % 2):
            dpg.add_text(str(moves_number // 2 + 1) + ") " + engine.chess_moves[moves_number - 1] + "\t\n")
        elif(moves_number > 0):
            dpg.add_text(str(moves_number // 2) + ") " + engine.chess_moves[moves_number - 2] + "\t" + engine.chess_moves[moves_number - 1] + "\n")
        else:
            dpg.add_text("No moves\tNo moves", parent= "moves_holder")

    if(engine.white_move):
        dpg.add_text("WHITE to move", tag= "to_move_holder", parent= "notation_Window")
    else:
        dpg.add_text("BLACK to move", tag= "to_move_holder", parent= "notation_Window")

    if(engine.enabled):
        dpg.add_text("Autoplay On", tag= "autoplay", parent= "notation_Window")

    if(engine.game_result != None):
        dpg.add_text(engine.game_result, tag= "over", parent= "evaluation_Window")

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
                    width = screenWidth * r1, height= screenHeight):
        
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
            dpg.add_button(label="New Board", callback= board_start)
            dpg.add_button(label="<-")
            dpg.add_button(label="->")

    # Set a handler for mouse clicks
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=mouse_clicked, user_data="board_Window")

    with dpg.window(label="notation window", tag="notation_Window", pos= (int(screenWidth * r1), 0), 
                    no_resize= True, no_collapse= True, no_close= True, no_move= True,
                    width = int(screenWidth * (1 - r1) * r2), height= int(screenHeight * r3)):
        # Create a scrollable child window to display the moves
        with dpg.child_window(width = int(screenWidth * (1 - r1) * (r2 - 0.03)), height= int(screenHeight * (r3 - 0.1)) , 
                              border=True, tag= "notation"):
            # Display the formatted chess moves
            dpg.add_text("  WHITE\tBLACK\n", parent= "notation")
            dpg.add_text("No moves\tNo moves", tag = "moves_holder", parent= "notation")
            dpg.add_text("WHITE to move", tag= "to_move_holder", parent= "notation_Window")

    with dpg.window(label="engine window", tag="engine_Window", pos= (int(screenWidth * (r1 + r2 - r1 * r2)), 0), 
                    no_resize= True, no_collapse= True, no_close= True, no_move= True,
                    width = int(screenWidth * (1 - r1) * (1 - r2)), height= int(screenHeight * r3)):
        
        with dpg.group(tag= "engine1_param"):
            dpg.add_text("Engine 1 parameter (White / autoplay):")
            dpg.add_button(label= "Select Bot", tag= 'engine1', callback= bot_selection)
            # Add an integer slider for timer
            dpg.add_slider_int(label="(secs) process", tag= "eng_timer1", default_value=10, min_value=10, max_value=120)

        with dpg.group(tag= "engine2_param", pos= (0, int(screenHeight * r3 / 2))):
            dpg.add_text("Engine 2 parameter (Black):")
            dpg.add_button(label= "Select Bot", tag= 'engine2', callback= bot_selection)
            # Add an integer slider for timer
            dpg.add_slider_int(label="(secs) process", tag= "eng_timer2", default_value=10, min_value=10, max_value=120)


    with dpg.window(label="evaluation window", tag="evaluation_Window", pos= (int(screenWidth * r1), int(screenHeight * r3)), 
                    no_resize= True, no_collapse= True, no_close= True, no_move= True,
                    width = int(screenWidth * (1 - r1)), height= int(screenWidth * (1 - r3))):       
        with dpg.group(tag = "evaluation_buttons", horizontal= True):
            dpg.add_button(label= "random_move", callback= engine_random)
            dpg.add_button(label= "|>", callback= engine_auto)
            dpg.add_button(label= "analyse")

        with dpg.group(tag = "position_buttons", pos= (ofs, int(screenHeight * r3 / 4))):
            dpg.add_text("Position:")
            with dpg.group(tag= "position_setup", horizontal= True):
                dpg.add_input_text(tag="string_input", hint="fen notation")
                dpg.add_button(label="setup", callback=submit_Fen)

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
dpg.create_viewport(title="chess", 
                    width= screenWidth, height= screenHeight, 
                    max_width= screenWidth, max_height= screenHeight, resizable=False)

# Setup and start the Dear PyGui rendering loop
dpg.setup_dearpygui()
dpg.show_viewport()

#dpg.start_dearpygui()      
while dpg.is_dearpygui_running():
    #print("this will run every frame")

    engine_automove()

    dpg.render_dearpygui_frame()
    #dpg.stop_dearpygui()

# Cleanup Dear PyGui context after use
dpg.destroy_context()