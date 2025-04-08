import chess
import random

class ChessianEngine:
    def __init__(self):
        self.name = "Chessian"
        self.author = "Ruby"
        self.board = chess.Board()

    def get_random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)

    def evaluate_position(self):
        # Always return 42 centipawns (0.42 pawns)
        return 42

def main():
    engine = ChessianEngine()
    
    while True:
        cmd = input().strip()
        
        if cmd == "quit":
            break
            
        elif cmd == "uci":
            print(f"id name {engine.name}")
            print(f"id author {engine.author}")
            print("uciok")
            
        elif cmd == "isready":
            print("readyok")
            
        elif cmd == "ucinewgame":
            engine.board.reset()
            
        elif cmd.startswith("position"):
            parts = cmd.split()
            if len(parts) >= 2:
                if parts[1] == "startpos":
                    engine.board.reset()
                    if len(parts) > 3 and parts[2] == "moves":
                        for move in parts[3:]:
                            engine.board.push_uci(move)
                            
        elif cmd.startswith("go"):
            move = engine.get_random_move()
            score = engine.evaluate_position()
            print(f"info score cp {score}")
            print(f"bestmove {move.uci()}")

if __name__ == "__main__":
    main()