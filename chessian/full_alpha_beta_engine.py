"""
full moves alpha beta engine, based on evaluation only
"""
import chess
MATE_SCORE = 1000000


class FullAlphaBetaEngine:
    """
    Engine based on alpha-beta pruning with iterative deepening.
    """
    def __init__(self, board, heuristic, depth) -> None:
        """
        heuristic: object with methods 'evaluate_position'.
        depth: maximum depth of the search tree
        max_nodes: maximum number of nodes to evaluate
        """
        self.name = "AlphaBetaEngine"
        self.board = board
        self.heuristic = heuristic
        self.depth = depth
        self.max_nodes = float('inf')
        self._best_move = None
        self._best_score = None
        self.nodes_evaluated = 0

    def _alpha_beta(self, depth, alpha, beta) -> tuple[chess.Move, float]:
        """
        depth: remaining depth to search
        alpha: alpha value for pruning (max score for maximizing player)
        beta: beta value for pruning (min score for minimizing player)
        """
        self.nodes_evaluated += 1
        if self.nodes_evaluated >= self.max_nodes:
            return None, 0
        
        if self.board.is_checkmate():
            return None, -MATE_SCORE if self.board.turn == chess.WHITE else MATE_SCORE
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return None, 0

        if depth == 0:
            return None, self.heuristic.evaluate_position(self.board)

        moves = list(self.board.legal_moves)
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
    
    def get_best_move(self, max_nodes):
        import random
        self._best_move = random.choice(list(self.board.legal_moves))
        self._best_score = None
        self.max_nodes = max_nodes
        self.nodes_evaluated = 0
        
        # Iterative deepening
        for current_depth in range(1, self.depth + 1):
            self.nodes_evaluated = 0
            move, score = self._alpha_beta(current_depth, -float('inf'), float('inf'))
            
            # If we hit the node limit, use the last complete iteration
            if self.nodes_evaluated >= self.max_nodes:
                break
                
            self._best_move = move
            self._best_score = score
            
        return self._best_move

    def evaluate_position(self):
        if self._best_score is None:
            return 0
        return self._best_score
