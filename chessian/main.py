import chess
import random
import math
import sys
from time import time

class Node:
    def __init__(self, board, move=None, parent=None):
        self.board = board.copy()
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)

class ChessianEngine:
    def __init__(self, board, mode):
        self.name = "Chessian"
        self.author = "Ruby"
        self.mode = mode
        self.board = board
        self.exploration_constant = 10

    def get_random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)

    def get_best_move(self, time_limit=1.0):
        if self.mode == 'random':
            return self.get_random_move()
        
        # MCTS implementation
        root = Node(self.board)
        end_time = time() + time_limit

        while time() < end_time:
            node = root
            board = self.board.copy()

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = self.uct_select_child(node)
                board.push(node.move)

            # Expansion
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                board.push(move)
                node = self.add_child(node, move, board)

            # Simulation
            while not board.is_game_over():
                board.push(random.choice(list(board.legal_moves)))

            # Backpropagation
            result = self.get_result(board)
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent

        return sorted(root.children, key=lambda c: c.visits)[-1].move

    def evaluate_position(self):
        return 42  # Placeholder for a real evaluation function 

    def uct_select_child(self, node):
        exploitation = lambda c: c.wins / c.visits if node.board.turn == chess.WHITE else (c.visits - c.wins) / c.visits
        exploration = lambda c: math.sqrt(2 * math.log(node.visits) / c.visits)
        uct = lambda c: exploitation(c) + self.exploration_constant * exploration(c)
        return max(node.children, key=uct)

    def add_child(self, parent, move, board):
        child_node = Node(board, move, parent)
        parent.untried_moves.remove(move)
        parent.children.append(child_node)
        return child_node

    def get_result(self, board):
        if board.is_checkmate():
            return 1 if board.turn == chess.WHITE else 0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0.5
        else:
            return 0

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'random'
    engine = ChessianEngine(chess.Board(), mode)
    
    while True:
        cmd = input().strip()
        
        if cmd == "quit":
            break
            
        elif cmd == "uci":
            print(f"id name {engine.name} ({mode})")
            print(f"id author {engine.author}")
            print("uciok")
            
        elif cmd == "isready":
            print("readyok")
            
        elif cmd == "ucinewgame":
            engine.board.reset()
            
        elif cmd.startswith("position"):
            board = chess.Board()
            parts = cmd.split()
            if len(parts) >= 2:
                if parts[1] == "startpos":
                    board.reset()
                    if len(parts) > 3 and parts[2] == "moves":
                        for move in parts[3:]:
                            board.push_uci(move)
                elif parts[1] == "fen":
                    fen = " ".join(parts[2:8])  # Join the FEN parts
                    board.set_fen(fen)
                    if len(parts) > 9 and parts[8] == "moves":
                        for move in parts[9:]:
                            board.push_uci(move)
            engine = ChessianEngine(board, mode)
                            
        elif cmd.startswith("go"):
            # Extract movetime if specified, default to 1 second
            movetime = 1.0  # Default time in seconds
            parts = cmd.split()
            if 'movetime' in parts:
                try:
                    # UCI movetime is in milliseconds, convert to seconds
                    movetime_index = parts.index('movetime') + 1
                    movetime = float(parts[movetime_index]) / 1000.0
                except (IndexError, ValueError):
                    pass
            
            move = engine.get_best_move(time_limit=movetime)
            score = engine.evaluate_position()
            print(f"info score cp {score}")
            print(f"bestmove {move.uci()}")

if __name__ == "__main__":
    main()