"""
Abstract Bot Interface with Threading and Multiprocessing Support
Provides clean separation between different bot implementations
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
import chess
from chess import Board
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
from dataclasses import dataclass
from enum import Enum


class BotType(Enum):
    """Types of bot implementations"""
    RANDOM = "random"
    NEURAL_NET = "neural_net"
    MINIMAX = "minimax"
    MCTS = "mcts"


@dataclass
class MoveResult:
    """Result from bot calculation"""
    move: Optional[chess.Move]
    evaluation: Optional[float] = None
    depth: Optional[int] = None
    nodes_searched: Optional[int] = None
    time_taken: float = 0.0
    pv_line: Optional[list] = None  # Principal variation


class AbstractBot(ABC):
    """
    Abstract base class for all chess bots
    Provides interface for move calculation with threading/multiprocessing support
    """
    
    def __init__(self, name: str, bot_type: BotType):
        """
        Initialize the bot
        
        Args:
            name: Human-readable name for the bot
            bot_type: Type of bot implementation
        """
        self.name = name
        self.bot_type = bot_type
        self._thinking = False
        self._stop_thinking = False
        self._calculation_thread = None
        
    @abstractmethod
    def _calculate_move_sync(self, board: Board, time_limit: float) -> MoveResult:
        """
        Synchronous move calculation - must be implemented by subclasses
        This is the core algorithm implementation
        
        Args:
            board: Current board state
            time_limit: Maximum time in seconds for calculation
            
        Returns:
            MoveResult with move and optional analysis
        """
        pass
    
    def calculate_move_async(self, board: Board, time_limit: float, 
                            callback: Optional[Callable] = None) -> threading.Thread:
        """
        Calculate move asynchronously in a separate thread
        
        Args:
            board: Current board state
            time_limit: Maximum time for calculation
            callback: Optional callback(result: MoveResult) called when complete
            
        Returns:
            Thread object for the calculation
        """
        def _async_wrapper():
            self._thinking = True
            self._stop_thinking = False
            try:
                result = self._calculate_move_sync(board.copy(), time_limit)
                if callback and not self._stop_thinking:
                    callback(result)
            except Exception as e:
                print(f"Error in bot calculation: {e}")
                if callback:
                    callback(MoveResult(move=None))
            finally:
                self._thinking = False
        
        thread = threading.Thread(target=_async_wrapper, daemon=True)
        self._calculation_thread = thread
        thread.start()
        return thread
    
    def stop_calculation(self):
        """Stop any ongoing calculation"""
        self._stop_thinking = True
        if self._calculation_thread and self._calculation_thread.is_alive():
            self._calculation_thread.join(timeout=0.5)
    
    @property
    def is_thinking(self) -> bool:
        """Check if bot is currently calculating"""
        return self._thinking
    
    def should_stop(self) -> bool:
        """Check if calculation should be stopped"""
        return self._stop_thinking


class ProcessBot(AbstractBot):
    """
    Bot that uses multiprocessing for move calculation
    Useful for CPU-intensive algorithms or when you need true parallelism
    """
    
    def __init__(self, name: str, bot_type: BotType):
        super().__init__(name, bot_type)
        self._process = None
        self._result_queue = None
    
    @abstractmethod
    def _calculate_move_process(self, fen: str, time_limit: float, 
                               result_queue: Queue) -> None:
        """
        Process-based move calculation
        Must be implemented by subclasses
        
        Args:
            fen: Board position as FEN string
            time_limit: Maximum time for calculation
            result_queue: Queue to put result into
        """
        pass
    
    def calculate_move_async(self, board: Board, time_limit: float,
                            callback: Optional[Callable] = None) -> Process:
        """
        Calculate move in a separate process
        
        Args:
            board: Current board state
            time_limit: Maximum time for calculation
            callback: Optional callback(result: MoveResult) called when complete
            
        Returns:
            Process object for the calculation
        """
        self._thinking = True
        self._stop_thinking = False
        self._result_queue = mp.Queue()
        
        def _monitor_process():
            """Monitor the process and call callback when done"""
            try:
                # Wait for result with timeout
                if self._result_queue:
                    result_dict = self._result_queue.get(timeout=time_limit + 1.0)
                    result = MoveResult(**result_dict)
                    if callback and not self._stop_thinking:
                        callback(result)
            except Exception as e:
                print(f"Error in process calculation: {e}")
                if callback:
                    callback(MoveResult(move=None))
            finally:
                self._thinking = False
                if self._process and self._process.is_alive():
                    self._process.terminate()
        
        # Start the calculation process
        self._process = Process(
            target=self._calculate_move_process,
            args=(board.fen(), time_limit, self._result_queue),
            daemon=True
        )
        self._process.start()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=_monitor_process, daemon=True)
        monitor_thread.start()
        
        return self._process
    
    def stop_calculation(self):
        """Stop any ongoing calculation"""
        self._stop_thinking = True
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)


# ============ Concrete Bot Implementations ============

class MinimaxBot(ProcessBot):
    """
    Minimax bot using multiprocessing
    Example of CPU-intensive algorithm using separate process
    """
    
    def __init__(self, max_depth: int = 4):
        super().__init__(f"Minimax (depth {max_depth})", BotType.MINIMAX)
        self.max_depth = max_depth
    
    def _calculate_move_sync(self, board: Board, time_limit: float) -> MoveResult:
        """Synchronous minimax calculation"""
        start_time = time.time()
        nodes = [0]  # Counter for nodes searched
        
        def minimax(board: Board, depth: int, alpha: float, beta: float, 
                   maximizing: bool) -> float:
            """Standard minimax with alpha-beta pruning"""
            if self.should_stop():
                return 0.0
            
            nodes[0] += 1
            
            if depth == 0 or board.is_game_over():
                return self._evaluate_position(board)
            
            if time.time() - start_time > time_limit:
                return self._evaluate_position(board)
            
            legal_moves = list(board.legal_moves)
            
            if maximizing:
                max_eval = float('-inf')
                for move in legal_moves:
                    board.push(move)
                    eval = minimax(board, depth - 1, alpha, beta, False)
                    board.pop()
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for move in legal_moves:
                    board.push(move)
                    eval = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval
        
        # Find best move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return MoveResult(move=None)
        
        best_move = None
        best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in legal_moves:
            if self.should_stop():
                break
            
            board.push(move)
            eval = minimax(board, self.max_depth - 1, float('-inf'), 
                          float('inf'), board.turn == chess.WHITE)
            board.pop()
            
            if board.turn == chess.WHITE:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move
        
        time_taken = time.time() - start_time
        
        return MoveResult(
            move=best_move,
            evaluation=best_eval,
            depth=self.max_depth,
            nodes_searched=nodes[0],
            time_taken=time_taken
        )
    
    def _calculate_move_process(self, fen: str, time_limit: float, 
                               result_queue: Queue) -> None:
        """Process-based calculation"""
        board = Board(fen)
        result = self._calculate_move_sync(board, time_limit)
        
        # Convert result to dict for queue
        result_dict = {
            'move': result.move,
            'evaluation': result.evaluation,
            'depth': result.depth,
            'nodes_searched': result.nodes_searched,
            'time_taken': result.time_taken
        }
        result_queue.put(result_dict)
    
    def _evaluate_position(self, board: Board) -> float:
        """Simple material evaluation"""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        return score


# ============ Bot Factory ============

# def create_bot(bot_type: str, **kwargs) -> AbstractBot:
#     """
#     Factory function to create different bot types
    
#     Args:
#         bot_type: Type of bot ("random", "neural_net", "minimax")
#         **kwargs: Additional arguments for specific bot types
        
#     Returns:
#         Bot instance
#     """
#     bot_type = bot_type.lower()
    
#     if bot_type == "random":
#         return RandomBot()
#     elif bot_type == "neural_net":
#         model_name = kwargs.get("model_name", "simple_bot")
#         return NeuralNetBot(model_name)
#     elif bot_type == "minimax":
#         depth = kwargs.get("depth", 3)
#         return MinimaxBot(depth)
#     else:
#         print(f"Unknown bot type '{bot_type}', using random")
#         return RandomBot()
