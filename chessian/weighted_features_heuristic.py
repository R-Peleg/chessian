"""
Heuristic usign linear combination of features to evaluate the position.
"""
import chess


class WeightedFeaturesHeuristic:
    def __init__(self, feature_evaluator, weights: dict[str, float]):
        self.feature_evaluator = feature_evaluator
        self.weights = weights
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the position using a linear combination of features.
        
        Args:
            board: The chess board to evaluate.
        
        Returns:
            The evaluation score for the position.
        """
        features = self.feature_evaluator.get_features(board)
        score = 0.0
        for feature, value in features.items():
            if feature in self.weights:
                score += self.weights[feature] * value
        return score

    def top_k_moves(self, board: chess.Board, k: int) -> list:
        raise NotImplementedError("Top-k moves not implemented for WeightedFeaturesHeuristic.")
