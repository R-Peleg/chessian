import chess
import random

class RandomEvaluator:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def evaluate_position(self, board: chess.Board) -> float:
       return random.gauss(self.mean, self.stddev)

    def top_k_moves(self, board: chess.Board, k: int) -> list:
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        return legal_moves[:k] if legal_moves else []
