import chess
import sys
from evaluator import Evaluator
from random_engine import RandomEngine
from mcts_engine import MCTSEngine
from direct_engine import DirectEngine
from llm_evaluator import LLMEvaluator


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'random'
    board = chess.Board()
    evaluator = Evaluator()
    llm_evaluator = LLMEvaluator('Qwen/Qwen2.5-0.5B', 'cpu')
    engines = {
        'random': RandomEngine(board, evaluator),
        'mcts': MCTSEngine(board, evaluator),
        'llm_mcts': MCTSEngine(board, llm_evaluator),
    }
    engine = engines.get(mode, RandomEngine(board, evaluator))
    
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
            engine = RandomEngine(board, evaluator) if mode == 'random' else MCTSEngine(board, evaluator)
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
            engine = RandomEngine(board, evaluator) if mode == 'random' else MCTSEngine(board, evaluator)
        elif cmd.startswith("go"):
            movetime = 1.0
            parts = cmd.split()
            if 'movetime' in parts:
                try:
                    movetime_index = parts.index('movetime') + 1
                    movetime = float(parts[movetime_index]) / 1000.0
                except (IndexError, ValueError):
                    pass
            move = engine.get_best_move(time_limit=movetime)
            score = engine.evaluate_position() if hasattr(engine, 'evaluate_position') else 0
            print(f"info score cp {score}")
            print(f"bestmove {move.uci()}")

if __name__ == "__main__":
    main()