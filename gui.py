"""
Chess GUI using DearPyGUI
Refactored for better performance and maintainability
"""

import dearpygui.dearpygui as dpg
import os
import chess
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Import your existing modules
from engine import AbstractEngine, GameResult
import piece_displayer as p_display

class ClickState(Enum):
    """Enumeration for click states"""
    NONE = 0
    FIRST_CLICK = 1
    SECOND_CLICK = 2
    PROMOTION = 3


@dataclass
class BoardSettings:
    """Configuration for board display"""
    board_square: int = 8
    square_size: int = 75
    piece_scale: float = 0.55
    offset: int = 10
    
    @property
    def board_size(self) -> int:
        return self.board_square * self.square_size

@dataclass
class ScreenSettings:
    """Configuration for screen layout"""
    width: int = 1280
    height: int = 720
    board_ratio: float = 0.5
    notation_ratio: float = 0.3
    eval_ratio: float = 0.4

###########################################################
class ChessGUI:
    """
    ChessGUI class
    """
    
    def __init__(self, engine: AbstractEngine):
        self.engine = engine
        self.board_settings = BoardSettings()
        self.screen_settings = ScreenSettings()
        
        # UI State
        self.flipped = False
        self.click_state = ClickState.NONE
        self.selected_square = -1
        self.move_notation = ""
        self.hold_piece = ""
        
        # Piece textures
        self.piece_textures = {}

        # UI element tags for cleanup
        self.temp_elements = set()
        
        # Initialize piece slicer
        self._initialize_pieces()
        
        # Set up engine observer
        self.engine.add_observer(self._on_engine_event)
    
    def _initialize_pieces(self):
        """Initialize piece images"""
        if not os.path.exists(p_display.directory):
            os.makedirs(p_display.directory, exist_ok=True)
            p_display.slice_pieces()
    
    def _on_engine_event(self, event: str, engine: AbstractEngine, **kwargs):
        """Handle engine state changes"""
        if event in ["move_made", "reset", "position_set", "move_undone"]:
            self._update_board_display()
            self._update_move_notation()
        elif event in ["autoplay_toggled", "autoplay_set"]:
            self._update_autoplay_display()
    
    # ============ Texture Management ============
    
    def _load_piece_texture(self, image_path: str, tag: str):
        """Load a piece image as texture"""
        if tag not in self.piece_textures:
            try:
                width, height, channels, image_data = dpg.load_image(image_path)
                with dpg.texture_registry(show=False):
                    dpg.add_static_texture(width, height, image_data, tag=tag)
                self.piece_textures[tag] = (width, height)
            except Exception as e:
                print(f"Failed to load texture {tag}: {e}")
    
    def _convert_piece_symbol(self, symbol: str) -> str:
        """Convert chess piece symbol to texture tag"""
        color_name = p_display.color_name[0] if symbol.islower() else p_display.color_name[1]
        piece_map = {
            'k': 'King', 'q': 'Queen', 'r': 'Rook', 
            'b': 'Bishop', 'n': 'Knight', 'p': 'Pawn'
        }
        return color_name + piece_map[symbol.lower()]
    
    # ============ Coordinate Conversion ============
    
    def _screen_to_board(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert screen coordinates to board coordinates"""
        x, y = screen_pos
        file = int(x / self.board_settings.square_size)
        rank = int(y / self.board_settings.square_size)
        
        if self.flipped:
            file = 7 - file
        else:
            rank = 7 - rank
            
        return file, rank
    
    def _board_to_screen(self, board_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates"""
        file, rank = board_pos
        
        if self.flipped:
            screen_file = 7 - file
            screen_rank = rank
        else:
            screen_file = file
            screen_rank = 7 - rank
            
        return (screen_file * self.board_settings.square_size,
                screen_rank * self.board_settings.square_size)
    
    def _is_valid_click(self, pos: Tuple[int, int]) -> bool:
        """Check if click position is within board bounds"""
        x, y = pos
        return (self.board_settings.offset <= x <= self.board_settings.board_size + self.board_settings.offset and
                self.board_settings.offset <= y <= self.board_settings.board_size + self.board_settings.offset)
    
    # ============ Move Input Handling ============
    
    def _handle_mouse_click(self, sender, app_data, user_data):
        """Handle mouse clicks on the board"""
        # Get mouse position relative to board window
        mouse_pos = dpg.get_mouse_pos()
        window_pos = dpg.get_item_pos(user_data)
        window_size = dpg.get_item_rect_size(user_data)
        
        # Check if click is within board window
        if not (window_pos[0] <= mouse_pos[0] <= window_pos[0] + window_size[0] and
                window_pos[1] <= mouse_pos[1] <= window_pos[1] + window_size[1]):
            return
        
        # Get relative position
        rel_pos = (mouse_pos[0] - window_pos[0], mouse_pos[1] - window_pos[1])
        
        # Check if click is on board
        if not self._is_valid_click(rel_pos):
            return
        
        # Convert to board coordinates
        file, rank = self._screen_to_board(rel_pos)
        square = chess.square(file, rank)
        
        self._process_square_click(square, rel_pos)
    
    def _process_square_click(self, square: int, screen_pos: Tuple[int, int]):
        """Process a click on a specific square"""
        if self.click_state == ClickState.NONE:
            self._handle_first_click(square, screen_pos)
        elif self.click_state == ClickState.FIRST_CLICK:
            self._handle_second_click(square, screen_pos)
        elif self.click_state == ClickState.PROMOTION:
            self._handle_promotion_click(square, screen_pos)
    
    def _handle_first_click(self, square: int, screen_pos: Tuple[int, int]):
        """Handle first click of move input"""
        piece = self.engine.get_piece_at(square)
        if piece and ((piece.color == chess.WHITE and self.engine.current_turn) or
                     (piece.color == chess.BLACK and not self.engine.current_turn)):
            self.selected_square = square
            self.move_notation = chess.square_name(square)
            self.hold_piece = piece.symbol()
            self.click_state = ClickState.FIRST_CLICK
            self._highlight_square(screen_pos, p_display.highlight_color, self.hold_piece)  # highlight color
    
    def _handle_second_click(self, square: int, screen_pos: Tuple[int, int]):
        """Handle second click of move input"""
        if square == self.selected_square:
            # Clicked same square - cancel move
            self._reset_move_input()
            return
        
        self.move_notation += chess.square_name(square)
        
        # Check for pawn promotion
        piece = self.engine.get_piece_at(self.selected_square)
        if (piece and piece.piece_type == chess.PAWN and
            ((piece.color == chess.WHITE and chess.square_rank(square) == 7) or
             (piece.color == chess.BLACK and chess.square_rank(square) == 0))):
            self.click_state = ClickState.PROMOTION
            self._show_promotion_dialog(screen_pos)
            return
        
        self._attempt_move()
    
    def _handle_promotion_click(self, square: int, screen_pos: Tuple[int, int]):
        """Handle promotion piece selection"""
        promotion_pieces = ('q', 'r', 'b', 'n')

        # Determine if click is within promotion dialog area
        file, rank = self._screen_to_board(screen_pos)
        if 3 <= file <= 4 and 3 <= rank <= 4:
            # Piece index coordinates
            piece_index = (file - 3) + 2 * (4 - rank)
            if(self.flipped):
                piece_index = (4 - file) + 2 * (rank - 3)

            if 0 <= piece_index < len(promotion_pieces):
                promotion_piece = promotion_pieces[piece_index]
                self.move_notation += '=' + promotion_piece.upper()
                self._attempt_move()
        else:
            # Clicked outside promotion dialog - cancel
            self._reset_move_input()
    
    ## TODO proper move display
    def _attempt_move(self):
        """Attempt to make the current move"""
        self.engine.make_move_san(self.move_notation)
        self._reset_move_input()
    
    def _reset_move_input(self):
        """Reset move input state"""
        self.click_state = ClickState.NONE
        self.selected_square = -1
        self.move_notation = ""
        self.hold_piece = ""
        self._clear_temporary_elements()
    
    # ============ Visual Feedback ============
    
    def _highlight_square(self, screen_pos: Tuple[int, int], color: Tuple[int, int, int], piece_type: Optional[str]= None):
        """Highlight a square with given color"""
        self._clear_temporary_elements()

        x, y = screen_pos
        square_x = (x // self.board_settings.square_size) * self.board_settings.square_size
        square_y = (y // self.board_settings.square_size) * self.board_settings.square_size
        
        tag = "temp_highlight"
        with dpg.draw_layer(tag=tag, parent= "board_squares"):
            dpg.draw_rectangle(
                pmin=(square_x, square_y),
                pmax=(square_x + self.board_settings.square_size, square_y + self.board_settings.square_size),
                fill=color,
            )

            # Draw piece also on the square
            if(piece_type):
                texture_tag = self._convert_piece_symbol(piece_type)
                if texture_tag in self.piece_textures:
                    width, height = self.piece_textures[texture_tag]
                    scaled_w = int(width * self.board_settings.piece_scale)
                    scaled_h = int(height * self.board_settings.piece_scale)
                    dpg.draw_image(
                        texture_tag,
                        pmin=(square_x, square_y),
                        pmax=(square_x + scaled_w, square_y + scaled_h),
                    )

        self.temp_elements.add(tag)
    
    def _show_promotion_dialog(self, screen_pos: Tuple[int, int]):
        """Show promotion piece selection dialog"""
        tag_base = "promotion_"
        
        promotion_pieces = ('q', 'r', 'b', 'n')
        with dpg.draw_layer(tag=tag_base + "layer", parent="board_squares"):
            for i in range(2):
                for j in range(2):
                    x = (3 + j) * self.board_settings.square_size
                    y = (3 + i) * self.board_settings.square_size
                    
                    # Draw background rectangle
                    rect_tag = f"{tag_base}rect_{i}_{j}"
                    dpg.draw_rectangle(
                        pmin=(x, y),
                        pmax=(x + self.board_settings.square_size, y + self.board_settings.square_size),
                        fill=(0, 0, 255),
                        tag=rect_tag,
                        parent=tag_base + "layer"
                    )
                    
                    # Draw piece
                    piece_idx = i * 2 + j
                    if piece_idx < len(promotion_pieces):
                        piece_symbol = promotion_pieces[piece_idx]
                        if not self.engine.current_turn:  # Black promotion
                            piece_symbol = piece_symbol.lower()
                        else:
                            piece_symbol = piece_symbol.upper()
                        
                        texture_tag = self._convert_piece_symbol(piece_symbol)
                        if texture_tag in self.piece_textures:
                            width, height = self.piece_textures[texture_tag]
                            scaled_w = int(width * self.board_settings.piece_scale)
                            scaled_h = int(height * self.board_settings.piece_scale)
                            
                            piece_tag = f"{tag_base}piece_{i}_{j}"
                            dpg.draw_image(
                                texture_tag,
                                [x, y],
                                [x + scaled_w, y + scaled_h],
                                tag=piece_tag,
                                parent=tag_base + "layer"
                            )
        
        self.temp_elements.add(tag_base + "layer")
    
    def _clear_temporary_elements(self):
        """Clear all temporary visual elements"""
        for tag in list(self.temp_elements):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.temp_elements.clear()
    
    # ============ Board Display ============
    
    def _update_board_display(self):
        """Update the visual board display"""
        self._clear_pieces()
        self._draw_pieces()
    
    def _clear_pieces(self):
        """Clear existing piece visuals"""
        if dpg.does_item_exist("piece_layer"):
            dpg.delete_item("piece_layer")
    
    def _draw_pieces(self):
        """Draw all pieces on the board"""
        with dpg.draw_layer(tag="piece_layer", parent="board_squares"):
            for square in chess.SQUARES:
                piece = self.engine.get_piece_at(square)
                if piece:
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)
                    screen_pos = self._board_to_screen((file, rank))
                    
                    texture_tag = self._convert_piece_symbol(piece.symbol())
                    if texture_tag in self.piece_textures:
                        width, height = self.piece_textures[texture_tag]
                        scaled_w = int(width * self.board_settings.piece_scale)
                        scaled_h = int(height * self.board_settings.piece_scale)
                        
                        dpg.draw_image(
                            texture_tag,
                            screen_pos,
                            [screen_pos[0] + scaled_w, screen_pos[1] + scaled_h],
                            parent="piece_layer"
                        )
    
    def _create_board_squares(self):
        """Create the visual chessboard squares"""
        with dpg.drawlist(width=self.board_settings.board_size, 
                         height=self.board_settings.board_size, 
                         tag="board_squares"):
            for rank in range(8):
                for file in range(8):
                    color = (p_display.white_color if (rank + file) % 2 == 0 
                           else p_display.brown_color)
                    
                    x = file * self.board_settings.square_size
                    y = rank * self.board_settings.square_size
                    
                    dpg.draw_rectangle(
                        pmin=(x, y),
                        pmax=(x + self.board_settings.square_size, 
                              y + self.board_settings.square_size),
                        fill=color,
                        color=(0, 0, 0, 255)
                    )
    
    # ============ Move Notation Display ============
    
    def _update_move_notation(self):
        """Update the move notation display"""
        # Clear existing notation
        if dpg.does_item_exist("moves_content"):
            dpg.delete_item("moves_content")
        
        # Create new notation display
        with dpg.group(tag="moves_content", parent="notation_scroll"):
            formatted_moves = self.engine.get_formatted_moves()
            
            if not formatted_moves:
                dpg.add_text("No moves\tNo moves")
            else:
                for move_num, white_move, black_move in formatted_moves:
                    move_text = f"{move_num}) {white_move}\t{black_move if black_move else ''}"
                    dpg.add_text(move_text)
        
        # Update turn indicator
        if dpg.does_item_exist("turn_indicator"):
            dpg.delete_item("turn_indicator")
        
        turn_text = "WHITE to move" if self.engine.current_turn else "BLACK to move"
        dpg.add_text(turn_text, tag="turn_indicator", parent="notation_panel")
    
    def _update_autoplay_display(self):
        """Update autoplay status display"""
        if dpg.does_item_exist("autoplay_status"):
            dpg.delete_item("autoplay_status")
        
        if self.engine.autoplay_enabled:
            status_text = f"Autoplay: {self.engine.engine_color.name}"
            dpg.add_text(status_text, tag="autoplay_status", parent="notation_panel")
    
    # ============ Button Callbacks ============
    
    def _on_flip_board(self):
        """Handle board flip button"""
        self.flipped = not self.flipped
        self._update_coordinate_labels()
        self._update_board_display()
    
    def _on_reset_board(self):
        """Handle reset board button"""
        self.engine.reset()
        self._reset_move_input()
    
    def _on_undo_move(self):
        """Handle undo move button"""
        self.engine.undo_move()
        self._reset_move_input()
    
    def _on_engine_move(self):
        """Handle engine move button"""
        time_limit = dpg.get_value("engine_time_slider")
        self.engine.make_engine_move(time_limit)
    
    def _on_toggle_autoplay(self):
        """Handle toggle autoplay button"""
        self.engine.toggle_autoplay()
    
    def _on_set_position(self):
        """Handle FEN position setup"""
        fen = dpg.get_value("fen_input")
        success = self.engine.set_position(fen)
        if not success:
            # Could add error feedback here
            print("Invalid FEN string")
    
    def _update_coordinate_labels(self):
        """Update file and rank labels based on flip state"""
        # Clear existing labels
        if dpg.does_item_exist("file_labels"):
            dpg.delete_item("file_labels")
        if dpg.does_item_exist("rank_labels"):
            dpg.delete_item("rank_labels")
        
        # Create new labels
        files = "abcdefgh" if not self.flipped else "hgfedcba"
        dpg.add_text(f"\t{' \t \t'.join(files.upper())}  ", tag="file_labels", parent="board_display")
        
        with dpg.group(tag="rank_labels", parent="board_display"):
            ranks = range(1, 9) if not self.flipped else range(8, 0, -1)
            for i, rank in enumerate(ranks):
                y_pos = i * self.board_settings.square_size + self.board_settings.square_size // 2
                dpg.add_text(str(rank), pos=(self.board_settings.board_size + 15, y_pos))
    
    # ============ Window Creation ============
    
    def _create_board_window(self):
        """Create the main board window"""
        board_size = int(self.screen_settings.width * self.screen_settings.board_ratio)
        
        with dpg.window(label="Chess Board", tag="board_window", 
                       pos=(0, 0), width=board_size, height=self.screen_settings.height,
                       no_resize=True, no_collapse=True, no_close=True, no_move=True):
            
            with dpg.group(tag="board_display"):
                self._create_board_squares()
                self._draw_pieces()
                self._update_coordinate_labels()
            
            # Board controls
            with dpg.group(tag="board_controls", horizontal=True):
                dpg.add_button(label="Flip Board", callback= self._on_flip_board)
                dpg.add_button(label="Reset", callback= self._on_reset_board)
                dpg.add_button(label="Undo", callback= self._on_undo_move)
            
            # Mouse handler
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=self._handle_mouse_click, 
                                          user_data="board_window")
    
    def _create_notation_window(self):
        """Create the notation display window"""
        x_pos = int(self.screen_settings.width * self.screen_settings.board_ratio)
        width = int(self.screen_settings.width * (1 - self.screen_settings.board_ratio) * self.screen_settings.notation_ratio)
        height = int(self.screen_settings.height * self.screen_settings.eval_ratio)
        
        with dpg.window(label="Game Notation", tag="notation_window",
                       pos=(x_pos, 0), width=width, height=height,
                       no_resize=True, no_collapse=True, no_close=True, no_move=True):
            
            with dpg.group(tag="notation_panel"):
                dpg.add_text("WHITE\tBLACK")
                
                with dpg.child_window(width=width-20, height=height-100, 
                                    border=True, tag="notation_scroll"):
                    pass  # Content will be added by _update_move_notation()
            
            self._update_move_notation()
    
    def _create_engine_window(self):
        """Create the engine control window"""
        x_pos = int(self.screen_settings.width * (self.screen_settings.board_ratio + 
                                                 (1 - self.screen_settings.board_ratio) * self.screen_settings.notation_ratio))
        width = int(self.screen_settings.width * (1 - self.screen_settings.board_ratio) * (1 - self.screen_settings.notation_ratio))
        height = int(self.screen_settings.height * self.screen_settings.eval_ratio)
        
        with dpg.window(label="Engine Controls", tag="engine_window",
                       pos=(x_pos, 0), width=width, height=height,
                       no_resize=True, no_collapse=True, no_close=True, no_move=True):
            
            dpg.add_text("Engine Settings:")
            dpg.add_slider_float(label="Time (sec)", tag="engine_time_slider",
                               default_value=1.0, min_value=0.1, max_value=10.0)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Engine Move", callback= self._on_engine_move)
                dpg.add_button(label="Toggle Autoplay", callback= self._on_toggle_autoplay)
    
    def _create_position_window(self):
        """Create the position setup window"""
        x_pos = int(self.screen_settings.width * self.screen_settings.board_ratio)
        y_pos = int(self.screen_settings.height * self.screen_settings.eval_ratio)
        width = int(self.screen_settings.width * (1 - self.screen_settings.board_ratio))
        height = int(self.screen_settings.height * (1 - self.screen_settings.eval_ratio))
        
        with dpg.window(label="Position Setup", tag="position_window",
                       pos=(x_pos, y_pos), width=width, height=height,
                       no_resize=True, no_collapse=True, no_close=True, no_move=True):
                        
            # Game status
            dpg.add_text("Game Status:")
            
            # This will be updated by engine events
            if dpg.does_item_exist("game_status"):
                dpg.delete_item("game_status")
            
            result = self.engine.game_result
            if result != GameResult.IN_PROGRESS:
                status_text = {
                    GameResult.WHITE_WIN: "White Wins!",
                    GameResult.BLACK_WIN: "Black Wins!",
                    GameResult.DRAW: "Draw"
                }.get(result, "Game Over")
                dpg.add_text(status_text, tag="game_status")
            
            
            dpg.add_separator()
            # Fen position setup
            with dpg.group(horizontal= True):
                dpg.add_text("FEN Position:")
                dpg.add_input_text(tag="fen_input", hint="Enter FEN string", width=400)
                dpg.add_button(label="Set Position", callback= self._on_set_position)
    
    # ============ Main Loop Integration ============
    
    def update(self):
        """Update method to be called in main loop"""
        # Check if engine should make a move in autoplay
        if (self.engine.should_make_engine_move() and 
            self.click_state == ClickState.NONE):
            self._on_engine_move()
    
    # ============ Initialization ============
    
    def initialize(self):
        """Initialize the GUI"""
        dpg.create_context()

        # Load piece textures
        for row in range(p_display.piece_grid_height):
            for col in range(p_display.piece_grid_width):
                piece_name = p_display.color_name[row] + p_display.chess_name[col]
                image_path = os.path.join(p_display.directory, f"{piece_name}.png")
                self._load_piece_texture(image_path, piece_name)
        
        # Create UI windows
        self._create_board_window()
        self._create_notation_window()
        self._create_engine_window()
        #self._create_position_window()
        
        # Configure viewport
        dpg.create_viewport(title="Chess Engine", 
                          width=self.screen_settings.width, 
                          height=self.screen_settings.height,
                          resizable=False)
        
        dpg.setup_dearpygui()
        
        #dpg.show_metrics()
        #dpg.set_viewport_vsync(False)

        dpg.show_viewport()
    
    def run(self):
        """Run the GUI main loop"""
        self.initialize()
        
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.engine.remove_observer(self._on_engine_event)
        dpg.destroy_context()


# Main execution
if __name__ == "__main__":
    engine = AbstractEngine()
    
    # Create and run GUI
    gui = ChessGUI(engine)
    gui.run()