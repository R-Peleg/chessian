"""
Feature extraction from Stockfish 15.1 classical evaluation function.
Stockfish 15.1 can be downloaded from https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_15.1.zip
"""
import subprocess
import asyncio
import chess
import chess.engine


class StockfishEvalFeatures:
    """
    Class to extract features from Stockfish classical evaluation function.
    """

    def __init__(self, stockfish: chess.engine.SimpleEngine) -> None:
        """
        Initialize the StockfishEvalFeatures class.

        Args:
            stockfish: An instance of the Stockfish engine.
        """
        self.stockfish = stockfish
        self.stockfish.configure({"Use NNUE": False})
        self.stockfish.configure({"Threads": 1})


    def get_features(self, board: chess.Board) -> dict[str, float]:
        """
        Get features from the board using Stockfish evaluation.

        Args:
            board: The chess board to evaluate.

        Returns:
            A dictionary of features extracted from the board.
        """
        class UciEvalCommand(chess.engine.BaseCommand[None]):
            def __init__(self, engine: chess.engine.UciProtocol) -> None:
                super().__init__(engine)
                self.engine = engine
                self.parse_state = 0
                self.features = {}

            def start(self) -> None:
                self.engine.send_line("eval")

            def line_received(self, line: str) -> None:
                # Format is 

                #  Contributing terms for the classical eval:
                # +------------+-------------+-------------+-------------+
                # |    Term    |    White    |    Black    |    Total    |
                # |            |   MG    EG  |   MG    EG  |   MG    EG  |
                # +------------+-------------+-------------+-------------+
                # |   Material |  ----  ---- |  ----  ---- | -2.01 -2.44 |
                # |  Imbalance |  ----  ---- |  ----  ---- |  0.54  0.38 |
                # |      Pawns |  0.29 -0.14 |  0.44  0.07 | -0.15 -0.22 |
                # |    Knights |  0.11 -0.01 |  0.02 -0.02 |  0.08  0.01 |
                # |    Bishops |  0.14 -0.07 |  0.00  0.00 |  0.14 -0.07 |
                # |      Rooks |  0.14  0.04 | -0.11 -0.04 |  0.25  0.08 |
                # |     Queens |  0.00  0.00 |  0.00  0.00 |  0.00  0.00 |
                # |   Mobility |  0.41  0.90 |  0.02  0.33 |  0.39  0.57 |
                # |King safety |  0.37 -0.06 | -0.12 -0.02 |  0.49 -0.04 |
                # |    Threats |  0.24  0.38 |  0.44  0.46 | -0.20 -0.07 |
                # |     Passed |  0.75  0.68 |  0.00  0.00 |  0.75  0.68 |
                # |      Space |  0.00  0.00 |  0.00  0.00 |  0.00  0.00 |
                # |   Winnable |  ----  ---- |  ----  ---- |  0.00  0.19 |
                # +------------+-------------+-------------+-------------+
                # |      Total |  ----  ---- |  ----  ---- |  0.28 -0.94 |
                # +------------+-------------+-------------+-------------+

                if line.count('+') == 5:
                    self.parse_state += 1
                    return
                if self.parse_state == 2:
                    parts = [p.strip() for p in line.split('|')]
                    feature_name = parts[1]
                    feature_values = parts[-2].split(' ')
                    feature_value = (float(feature_values[0]) + float(feature_values[-1])) / 2
                    self.features[feature_name] = feature_value
                if line.startswith('Final evaluation'):
                    self.result.set_result(self.features)
                    self.set_finished()
        self.stockfish.protocol._ucinewgame()
        self.stockfish.protocol._position(board)
        evaluate_output_coro = asyncio.wait_for(self.stockfish.protocol.communicate(UciEvalCommand), 10)
        evaluate_output_future = asyncio.run_coroutine_threadsafe(evaluate_output_coro, self.stockfish.protocol.loop)
        return evaluate_output_future.result()


def main():
    # Example usage
    import sys
    stockfish_path = sys.argv[1]
    fen = 'r3k1r1/3nqpPp/p2p2p1/1pp1p3/4N3/1PPPN1P1/1P3PBP/R3R1K1 b q - 3 19'
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as stockfish:
        stockfish_eval_features = StockfishEvalFeatures(stockfish)
        features = stockfish_eval_features.get_features(board)
    print(features)


if __name__ == "__main__":
    main()
