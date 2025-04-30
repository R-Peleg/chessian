"""
Combined feature extractor that aggregates features from multiple extractors.
"""
from typing import List
import chess


class CompositeFeatureExtractor:
    """
    A feature extractor that combines features from multiple extractors.
    """

    def __init__(self, extractors: List[object]) -> None:
        """
        Initialize with a list of feature extractors.

        Args:
            extractors: List of feature extractor objects that implement get_features method
        """
        self.extractors = extractors

    def get_features(self, board: chess.Board) -> dict[str, float]:
        """
        Get combined features from all extractors.

        Args:
            board: The chess board to evaluate.

        Returns:
            A dictionary containing all features from all extractors.
        """
        features = {}
        for extractor in self.extractors:
            features.update(extractor.get_features(board))
        return features


def main():
    # Example usage
    from stockfish_eval_features import StockfishEvalFeatures
    import chess.engine
    import sys

    stockfish_path = sys.argv[1]
    board = chess.Board()
    
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as stockfish:
        stockfish_features = StockfishEvalFeatures(stockfish)
        position_extractor = PositionFeatureExtractor([stockfish_features])
        features = position_extractor.get_features(board)
    print(features)


if __name__ == "__main__":
    main()
