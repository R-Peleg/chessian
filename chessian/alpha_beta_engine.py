import chess


class AlphaBetaEngine:
    """
    Engine based on top-k moves and alpha-beta pruning.
    """
    def __init__(self, board, heuristic, k, depth) -> None:
        """
        heuristic: object with methods 'evaluate_position' and 'top_k_moves'
        k: number of top moves to consider
        depth: depth of the search tree
        """
        self.name = "AlphaBetaEngine"
        self.board = board
        self.heuristic = heuristic
        self.k = k
        self.depth = depth
        self._best_move = None
        self._best_score = None
    
    def _alpha_beta(self, depth, alpha, beta) -> tuple[chess.Move, float]:
        """
        depth: remaining depth to search
        alpha: alpha value for pruning (max score for maximizing player)
        beta: beta value for pruning (min score for minimizing player)
        """
        if depth == 0:
            return None, self.heuristic.evaluate_position(self.board)

        moves = self.heuristic.top_k_moves(self.board, self.k)
        if not moves:
            return None, self.heuristic.evaluate_position(self.board)

        if self.board.turn == chess.WHITE:
            max_score = -float('inf')
            best_move = None
            for move in moves:
                self.board.push(move)
                _, score = self._alpha_beta(depth - 1, alpha, beta)
                self.board.pop()
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_move, max_score
        else:
            min_score = float('inf')
            best_move = None
            for move in moves:
                self.board.push(move)
                _, score = self._alpha_beta(depth - 1, alpha, beta)
                self.board.pop()
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_move, min_score
    
    def get_best_move(self, time_limit=1.0):
        if self._best_move is not None:
            return self._best_move

        best_move, best_score = self._alpha_beta(self.depth, -float('inf'), float('inf'))
        self._best_move = best_move
        self._best_score = best_score
        return best_move
    
    def evaluate_position(self):
        if self._best_score is not None:
            return self._best_score
        best_move, best_score = self._alpha_beta(self.depth, -float('inf'), float('inf'))
        self._best_move = best_move
        self._best_score = best_score
        return self._best_score
