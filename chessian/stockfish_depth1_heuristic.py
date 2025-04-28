import chess
import chess.engine


class StockfishDepth1Heuristic:
    """
    Heuristic based on depth=1 Stockfish evaluation
    """
    def __init__(self, stockfish_path: str):
        """
        Initialize the evaluator with the specified model and device.
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the position using Stockfish at depth 1.
        """
        result = self.engine.analyse(board, chess.engine.Limit(depth=1))
        score = result['score']
        if score.is_mate():
            return float('inf') if score.turn == chess.WHITE else float('-inf')
        return score.white().score()

    def top_k_moves(self, board: chess.Board, k: int) -> list:
        raise NotImplementedError()


def main():
    stockfish_path = r"C:\Users\ruby\chess\stockfish-windows-x86-64-avx2.exe"
    evaluator = StockfishDepth1Heuristic(stockfish_path)
    position = '8/p1B5/3P3k/1P6/r3K3/5R2/8/6R1 w - - 8 54'
    board = chess.Board(position)
    print("Position:", position)
    print("Evaluation:", evaluator.evaluate_position(board))

if __name__ == "__main__":
    main()
