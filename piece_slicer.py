from PIL import Image

piece_grid_width = 6  # 6 columns for white pieces and black pieces
piece_grid_height = 2  # 2 rows for white and black pieces

########## Change this for custom pieces ##########

# Board Colors
white_color = (255, 255, 255, 255)  # RGBA for white squares
brown_color = (150, 75, 0, 255)        # RGBA for black squares

# Size and layout for the pieces in the sprite sheet (assuming 6x2 grid of pieces)
piece_width = 120
piece_height = 120
color_name = ("Black", "White")
chess_name = ("Rook", "Bishop", "Queen", "King", "Knight", "Pawn")




offX = 0
offY = 28

# Specify the directory path
directory = "img"

# Path to the PNG file containing all the pieces
pieces_image_path = "CHESS PIECES.png"

########## ########## ########## ##########

# Load the sprite sheet
sprite_sheet = Image.open(pieces_image_path)

# Function to convert a Pillow image to a format suitable for Dear PyGui
def pil_to_image(pil_name, pil_image):
    with open(directory + "\\" + pil_name + ".png", "wb") as f:
        pil_image.save(f, "PNG")
        f.close()

# Function to slice the sprite sheet into individual pieces
def slice_pieces():
    for row in range(piece_grid_height):
        for col in range(piece_grid_width):
            # Calculate the position in the sprite sheet for each piece
            left = offX + col * piece_width
            top = offY * row + row * piece_height
            right = left + piece_width
            bottom = top + piece_height
            piece = sprite_sheet.crop((left, top, right, bottom))
            piece_name = color_name[row] + chess_name[col]  # For simplicity, name pieces based on their grid position
            pil_to_image(piece_name, piece)

    print("pieces sliced")