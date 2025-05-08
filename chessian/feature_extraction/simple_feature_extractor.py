import chess
import chess.engine


class SimpleFeatures:
    """
    Class to extract features from simple heuristics.
    """

    def get_features(self, board: chess.Board) -> dict[str, float]:
        """
        Get features from the board using Stockfish evaluation.

        Args:
            board: The chess board to evaluate.

        Returns:
            A dictionary of features extracted from the board.
        """
        material = 0
        for piece in board.piece_map().values():
            piece_value = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0,
            }[piece.piece_type]
            if piece.color == chess.WHITE:
                material += piece_value
            else:
                material -= piece_value
        board_copy = board.copy()
        if board_copy.turn != chess.WHITE:
            board_copy.push(chess.Move.null())
        mobility_white = len(list(board_copy.legal_moves))
        board_copy.push(chess.Move.null())
        mobility_black = len(list(board_copy.legal_moves))
        mobility = mobility_white - mobility_black
        return {
            "material_simple": material,
            "mobility_simple": mobility * 0.1,
        }


def main():
    fen = 'rnbq1rk1/1pppppbp/p5p1/8/P2P4/6P1/1PPNP1BP/RNBQ1RK1 b - - 0 8'
    board = chess.Board(fen)
    feature_extractor = SimpleFeatures()
    features = feature_extractor.get_features(board)
    print("Features:", features)


if __name__ == "__main__":
    main()
