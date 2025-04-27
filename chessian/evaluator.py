import chess
import random

class Evaluator:
    @staticmethod
    def evaluate_position(board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0 if board.turn else 1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        material_score = 0
        for piece_type, value in piece_values.items():
            material_score += len(board.pieces(piece_type, chess.WHITE)) * value
            material_score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return max(min(material_score, 10), -10)

    @staticmethod
    def sort_moves(board: chess.Board, moves) -> list:
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
    
    def top_k_moves(self, board: chess.Board, k: int) -> list:
        legal_moves = list(board.legal_moves)
        sorted_moves = self.sort_moves(board, legal_moves)
        return sorted_moves[:k] if sorted_moves else []
