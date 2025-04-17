import chess
import random
import math
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

class MCTSEngine:
    def __init__(self, board: chess.Board, evaluator, exploration_constant=1.5, 
                 node_budget=float('inf')):
        self.name = "Chessian MCTS"
        self.board = board
        self.evaluator = evaluator
        self.exploration_constant = exploration_constant
        self.root = Node(self.board)
        self.node_budget = node_budget
        self.node_count = 0

    def get_best_move(self, time_limit=1.0):
        end_time = time() + time_limit
        while time() < end_time and self.node_count < self.node_budget:
            node = self.root
            board = self.board.copy()

            # Selection
            while not node.untried_moves and node.children:
                node = self.uct_select_child(node)
                board.push(node.move)

            # Expansion
            if node.untried_moves:
                moves = self.evaluator.sort_moves(board, node.untried_moves)
                move = random.choice([moves[0]] * 5 + moves)
                board.push(move)
                node = self.add_child(node, move, board)

            # Simulation
            rollout_moves = 0
            while not board.is_game_over():
                moves = self.evaluator.sort_moves(board, board.legal_moves)
                if not moves:
                    break
                next_move = moves[0] if random.random() < 0.7 else random.choice(moves)
                board.push(next_move)
                rollout_moves += 1
                if rollout_moves > 5:
                    break
            result = self.evaluator.evaluate_position(board)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent
            self.node_count += 1

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
