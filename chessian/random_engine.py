import chess
import random

class RandomEngine:
    def __init__(self, board: chess.Board, evaluator):
        self.name = "Chessian Random"
        self.board = board
        self.evaluator = evaluator

    def get_best_move(self, time_limit=1.0):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)
    
    def evaluate_position(self):
        # Delegate to evaluator
        return self.evaluator.evaluate_position(self.board)
