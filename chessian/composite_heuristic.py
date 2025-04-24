import chess

class CompositeHeuristic:
    """
    Composite heuristic that delegates position evaluation and top-k moves to separate heuristics.
    """
    def __init__(self, pos_eval_heuristic, top_moves_heuristic):
        self.pos_eval_heuristic = pos_eval_heuristic
        self.top_moves_heuristic = top_moves_heuristic

    def evaluate_position(self, board: chess.Board) -> float:
        # Delegate position evaluation to the provided heuristic.
        return self.pos_eval_heuristic.evaluate_position(board)

    def top_k_moves(self, board: chess.Board, k: int) -> list:
        # Delegate top-k moves computation to the provided heuristic.
        return self.top_moves_heuristic.top_k_moves(board, k)
