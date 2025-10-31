"""
Abstract Chess Engine Interface
"""

from typing import Optional, List, Callable
from enum import Enum
import chess
import random

class GameResult(Enum):
    """Enumeration for game results"""
    IN_PROGRESS = "in_progress"
    WHITE_WIN = "white_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"


class EngineColor(Enum):
    """Enumeration for engine colors"""
    WHITE = chess.WHITE
    BLACK = chess.BLACK


class AbstractEngine:
    """
    Abstract base class for chess engines.
    Defines the interface that all engine implementations must follow.
    """
    
    def __init__(self):
        """Initialize the base engine"""
        self._board = chess.Board()
        self._move_history = []
        self._observers = []
        self._autoplay_enabled = False
        self._engine_color = EngineColor.WHITE
        
        self._thinking = False
    # ============ Core Game State Methods ============
    
    @property
    def board(self) -> chess.Board:
        """Get the current board state"""
        return self._board
    
    @property
    def fen(self) -> str:
        """Get the current position as FEN string"""
        return self._board.fen()
    
    @property
    def move_history(self) -> List[str]:
        """Get the list of moves in UCI notation"""
        return self._move_history.copy()
    
    @property
    def current_turn(self) -> bool:
        """Get current turn (True=White, False=Black)"""
        return self._board.turn == chess.WHITE
    
    @property
    def game_result(self) -> GameResult:
        """Get the current game result"""
        if self._board.is_checkmate():
            return GameResult.BLACK_WIN if self._board.turn == chess.WHITE else GameResult.WHITE_WIN
        elif (self._board.is_stalemate() or 
              self._board.is_insufficient_material() or 
              self._board.is_fivefold_repetition() or
              self._board.is_fifty_moves()):
            return GameResult.DRAW
        return GameResult.IN_PROGRESS
    
    @property
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_result != GameResult.IN_PROGRESS
    
    # ============ Board Management Methods ============
    
    def reset(self):
        """Reset the board to starting position"""
        self._board.reset()
        self._move_history.clear()
        self._autoplay_enabled = False
        self._thinking = False
        self._notify_observers("reset")
    
    def set_position(self, fen: str) -> bool:
        """
        Set board position from FEN string
        Returns True if successful, False otherwise
        """
        try:
            self.reset()
            self._board.set_fen(fen)
            self._notify_observers("position_set")
            return True
        except ValueError:
            return False
    
    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """Get piece at given square (0-63)"""
        return self._board.piece_at(square)
    
    # ============ Move Methods ============
    
    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move on the board
        Returns True if move was legal and executed, False otherwise
        """
        if move in self._board.legal_moves:
            self._move_history.append(self.get_move_san(move))
            self._board.push(move)
            self._notify_observers("move_made", move=move)
            return True
        return False
    
    def make_move_san(self, san: str) -> bool:
        """
        Make a move using SAN notation (e.g., 'e4', 'Nf3')
        Returns True if successful, False otherwise
        """
        try:
            move = self._board.parse_san(san)
            return self.make_move(move)
        except ValueError:
            return False
    
    def make_move_uci(self, uci: str) -> bool:
        """
        Make a move using UCI notation (e.g., 'e2e4')
        Returns True if successful, False otherwise
        """
        try:
            move = chess.Move.from_uci(uci)
            return self.make_move(move)
        except ValueError:
            return False
    
    def undo_move(self) -> bool:
        """
        Undo the last move
        Returns True if successful, False if no moves to undo
        """
        if len(self._board.move_stack) > 0:
            self._board.pop()
            if self._move_history:
                self._move_history.pop()
            self._notify_observers("move_undone")
            return True
        return False
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get list of all legal moves in current position"""
        return list(self._board.legal_moves)
    
    # ============ Engine Move Generation (Abstract) ============
    
    def calculate_best_move(self, time_limit: float = 1.0) -> Optional[chess.Move]:
        """
        Calculate the best move for the current position
        Must be implemented by subclasses
        
        Args:
            time_limit: Maximum time in seconds for calculation
            
        Returns:
            Best move or None if no legal moves
        """
        if self.is_game_over:
            return None

        import time
        time.sleep(5)
        return random.choice(self.get_legal_moves())
    
    

    def make_engine_move(self, time_limit: float = 1.0):
        """
        Calculate and make the engine's move
        Make current color's engine take move
        """
        if self.is_game_over:
            return None
            
        if(not self._thinking):
            self._thinking = True
            move = self.calculate_best_move(time_limit)
            if move:
                self.make_move(move)
                return move
        return None
    
    # ============ Autoplay Methods ============
    
    @property
    def autoplay_enabled(self) -> bool:
        """Check if autoplay is enabled"""
        return self._autoplay_enabled
    
    @property
    def engine_color(self) -> EngineColor:
        """Get the color the engine is playing"""
        return self._engine_color
    
    def toggle_autoplay(self):
        """Toggle autoplay on/off, setting engine color to current turn"""
        self._autoplay_enabled = not self._autoplay_enabled
        if self._autoplay_enabled:
            self._engine_color = EngineColor.WHITE if self.current_turn else EngineColor.BLACK
        self._notify_observers("autoplay_toggled", enabled=self._autoplay_enabled)
    
    def should_make_engine_move(self) -> bool:
        """Check if engine should make a move in current position"""
        if not self._autoplay_enabled or self.is_game_over:
            return False
        return ((self._engine_color == EngineColor.WHITE and self.current_turn) or
                (self._engine_color == EngineColor.BLACK and not self.current_turn))
    
    # ============ Observer Pattern ============
    
    def add_observer(self, callback: Callable):
        """Add an observer callback for state changes"""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def remove_observer(self, callback: Callable):
        """Remove an observer callback"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, event: str, **kwargs):
        """Notify all observers of a state change"""
        for observer in self._observers:
            observer(event, self, **kwargs)
    
    # ============ Utility Methods ============
    
    def get_move_san(self, move: chess.Move) -> str:
        """Convert a move to SAN notation"""
        return self._board.san(move)
    
    def is_check(self) -> bool:
        """Check if current position is check"""
        return self._board.is_check()
    
    def is_checkmate(self) -> bool:
        """Check if current position is checkmate"""
        return self._board.is_checkmate()
    
    def is_stalemate(self) -> bool:
        """Check if current position is stalemate"""
        return self._board.is_stalemate()
    
    def get_formatted_moves(self) -> List[tuple]:
        """
        Get formatted move pairs for display
        Returns list of tuples: [(move_num, white_move, black_move), ...]
        """
        formatted = []
        for i in range(0, len(self._move_history), 2):
            move_num = (i // 2) + 1
            white_move = self._move_history[i]
            black_move = self._move_history[i + 1] if i + 1 < len(self._move_history) else ""
            formatted.append((move_num, white_move, black_move))
        return formatted
