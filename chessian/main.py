import chess
import random
import math
import sys
from time import time


def evaluate_position(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0 if board.turn else 1.0  # Scale to [-1, 1] range
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0  # Draw
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King has no material value in this context
    }
    material_score = 0
    for piece_type, value in piece_values.items():
        material_score += len(board.pieces(piece_type, chess.WHITE)) * value
        material_score -= len(board.pieces(piece_type, chess.BLACK)) * value
    
    # Scale the material score to a reasonable range
    # The maximum theoretical material difference in chess is around 78 points
    # (8 pawns + 2 knights + 2 bishops + 2 rooks + 1 queen = 39 points per side)
    scaled_score = material_score / 78.0
    
    # Limit the range to [-0.95, 0.95] to leave room for terminal states
    return max(min(scaled_score, 0.95), -0.95)


def sort_moves(board: chess.Board, moves) -> list:
    """
    Sort moves based on their type and legality.
    """
    moves = list(moves)
    random.shuffle(moves)
    def move_score(move: chess.Move) -> float:
        score = 0
        if move.promotion:
            score += 10 + move.promotion * 3
        if board.is_capture(move):
            capture_piece = chess.PAWN if board.is_en_passant(move) else board.piece_at(move.to_square).piece_type
            score += 10 + capture_piece * 3
        if board.gives_check(move):
            score += 0.5
        if board.attackers(not board.turn, move.from_square):
            score += 0.1
        if board.attackers(not board.turn, move.to_square) and not board.attackers(board.turn, move.to_square):
            score -= 0.2
        return score

    sorted_moves = sorted(moves, key=move_score, reverse=True)
    return sorted_moves


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
        self.root = Node(self.board)

    def get_random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)

    def get_best_move(self, time_limit=1.0):
        if self.mode == 'random':
            return self.get_random_move()
        
        # MCTS implementation
        end_time = time() + time_limit

        while time() < end_time:
            node = self.root
            board = self.board.copy()

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = self.uct_select_child(node)
                board.push(node.move)

            # Expansion
            if node.untried_moves != []:
                moves = sort_moves(board, node.untried_moves)
                move = random.choice([moves[0]] * 5 + moves)
                board.push(move)
                node = self.add_child(node, move, board)

            # Simulation
            for _ in range(5):
                if board.is_game_over():
                    break
                moves = sort_moves(board, board.legal_moves)
                next_move = moves[0] if random.random() < 0.7 else random.choice(moves)
                board.push(next_move)
            result = evaluate_position(board)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent

        return max(self.root.children, key=lambda c: c.visits).move

    def evaluate_position(self):
        return round(100 * self.root.wins / self.root.visits) if self.root.visits > 0 else 0

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