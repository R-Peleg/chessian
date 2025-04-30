import chess
import sys
from evaluator import Evaluator
from random_engine import RandomEngine
from mcts_engine import MCTSEngine
from direct_engine import DirectEngine
from alpha_beta_engine import AlphaBetaEngine
from full_alpha_beta_engine import FullAlphaBetaEngine
# from llm_evaluator import LLMEvaluator
# from gemini_evaluator import GeminiEvaluator
from gemini_heuristic import GeminiHeuristic
from composite_heuristic import CompositeHeuristic
from weighted_features_heuristic import WeightedFeaturesHeuristic
from feature_extraction.stockfish_eval_features import StockfishEvalFeatures
from feature_extraction.gemini_eval_features import GeminiEvalFeatures
from feature_extraction.composite_features import CompositeFeatureExtractor

# Weights found by lasso regression
CLASSIC_FEATURES_WEIGHTS = {
    "Material": 0.6638062211120966,
    "Imbalance": 0.0,
    "Pawns": 0.9527885652092388,
    "Knights": 0.0,
    "Bishops": 0.14680968691215993,
    "Rooks": 0.0,
    "Queens": 0.0,
    "Mobility": 0.8271389978937992,
    "King safety": 0.45246437022494507,
    "Threats": 1.1228572943646506,
    "Passed": 0.7253141087156784,
    "Space": 0.0,
    "Winnable": 0.0
}

HYBRID_FEATURES_WEIGHTS = {
    "material": 0.0,
    "pawn_structure": 0.12149655354134242,
    "mobility": 0.3393729906043462,
    "king_safety": 0.0,
    "Material": 0.6296413122802713,
    "Imbalance": 0.0,
    "Pawns": 0.0,
    "Knights": 0.0,
    "Bishops": 0.0,
    "Rooks": 0.0,
    "Queens": 0.0,
    "Mobility": 0.0,
    "King safety": 0.019585003130121004,
    "Threats": 0.0,
    "Passed": 0.0,
    "Space": 0.0,
    "Winnable": 0.0,
}

def new_engine(mode, board):
    evaluator = Evaluator()
    # llm_evaluator = LLMEvaluator('Qwen/Qwen2.5-0.5B', 'cpu')
    # gem_heuristic = GeminiHeuristic('gemini-2.0-flash')
    # gem_moves_classic_eval = CompositeHeuristic(
    #     pos_eval_heuristic=WeightedFeaturesHeuristic(),
    #     top_moves_heuristic=gem_heuristic
    # )
    stockfish_path = "C:\\Users\\ruby\\chess\\stockfish-15-1.exe"
    if mode == 'ab_classic_features':
        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        return FullAlphaBetaEngine(board, WeightedFeaturesHeuristic(
            feature_evaluator=StockfishEvalFeatures(stockfish),
            weights=CLASSIC_FEATURES_WEIGHTS
        ), depth=10)
    elif mode == 'ab_hybrid_features':
        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        feature_extractor = CompositeFeatureExtractor([
            StockfishEvalFeatures(stockfish),
            GeminiEvalFeatures('gemma-3-27b-it')
        ])
        return FullAlphaBetaEngine(board, WeightedFeaturesHeuristic(
            feature_evaluator=feature_extractor,
            weights=HYBRID_FEATURES_WEIGHTS
        ), depth=10)
    engines = {
        'random': RandomEngine(board, evaluator),
        'mcts': MCTSEngine(board, evaluator),
        'alpha_beta': AlphaBetaEngine(board, evaluator, k=7, depth=4),
        'alpha_beta_gemini': AlphaBetaEngine(board, gem_heuristic, k=2, depth=3),
        # 'llm_mcts': MCTSEngine(board, llm_evaluator),
        'gemini_direct': DirectEngine(board, gem_heuristic),
        # 'gemini_mcts': MCTSEngine(board, GeminiEvaluator('gemini-2.0-flash')),
    }
    return engines.get(mode, RandomEngine(board, evaluator))


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'random'

    board = chess.Board()
    engine = new_engine(mode, board)

    while True:
        cmd = input().strip()
        
        if cmd == "quit":
            break
        elif cmd == "uci":
            print(f"id name {engine.name} ({mode})")
            print("id author Ruby")
            print("uciok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            board.reset()
            engine = new_engine(mode, board)
        elif cmd.startswith("position"):
            board = chess.Board()
            parts = cmd.split()
            if parts[1] == "startpos":
                board.reset()
                if len(parts) > 3 and parts[2] == "moves":
                    for move in parts[3:]:
                        board.push_uci(move)
            elif parts[1] == "fen":
                fen = " ".join(parts[2:8])
                board.set_fen(fen)
                if len(parts) > 9 and parts[8] == "moves":
                    for move in parts[9:]:
                        board.push_uci(move)
            engine = new_engine(mode, board)
        elif cmd.startswith("go"):
            nodes = 10_000
            parts = cmd.split()
            if 'nodes' in parts:
                try:
                    nodes_index = parts.index('nodes') + 1
                    nodes = int(parts[nodes_index])
                except (IndexError, ValueError):
                    pass
            
            move = engine.get_best_move(max_nodes=nodes)
            score = engine.evaluate_position() if hasattr(engine, 'evaluate_position') else 0
            print(f"info score cp {int(score * 100)}")
            print(f"bestmove {move.uci()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        path = sys.argv[0].replace('.py', '_exception.log')
        with open(path, 'a') as f:
            f.write(str(e) + '\n')
        raise