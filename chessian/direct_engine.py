import chess

class DirectEngine:
    def __init__(self, board: chess.Board, evaluator):
        self.name = "Chessian Direct"
        self.board = board
        self.evaluator = evaluator

    def get_best_move(self, time_limit=1.0):
        legal_moves = list(self.board.legal_moves)
        if hasattr(self.evaluator, 'top_k_moves'):
            best_moves = self.evaluator.top_k_moves(self.board, 1)
            if best_moves:
                return best_moves[0]
            else:
                return None
        sorted_moves = self.evaluator.sort_moves(self.board, legal_moves)
        return sorted_moves[0] if sorted_moves else None

    def evaluate_position(self):
        return self.evaluator.evaluate_position(self.board)
