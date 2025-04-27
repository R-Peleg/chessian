"""
from https://chatgpt.com/share/680e7013-4e84-8013-8194-d9f2ab08cd8b
"""

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000  # high value to avoid king captures
}

CENTER_SQUARES = [
    chess.D4, chess.D5, chess.E4, chess.E5,
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.F3, chess.F4, chess.F5, chess.F6
]

def is_endgame(board):
    """Simple endgame detection."""
    queen_count = len(list(board.pieces(chess.QUEEN, chess.WHITE))) + len(list(board.pieces(chess.QUEEN, chess.BLACK)))
    return queen_count <= 1 or bin(board.occupied_co[chess.WHITE]).count('1') + bin(board.occupied_co[chess.BLACK]).count('1') < 12

def material_score(board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    return score

def mobility_score(board):
    """Mobility = number of legal moves."""
    moves = list(board.legal_moves)
    return len(moves) * (5 if board.turn == chess.WHITE else -5)

def king_safety_score(board, color):
    """King safety = pawns around king."""
    king_square = board.king(color)
    if king_square is None:
        return 0

    king_rank = chess.square_rank(king_square)
    king_file = chess.square_file(king_square)

    safety = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)]
    for dr, df in directions:
        r, f = king_rank + dr, king_file + df
        if 0 <= r < 8 and 0 <= f < 8:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                safety += 15

    return safety

def center_control_score(board):
    """Control of center squares."""
    score = 0
    for sq in CENTER_SQUARES:
        attackers_white = board.attackers(chess.WHITE, sq)
        attackers_black = board.attackers(chess.BLACK, sq)
        score += 10 * (len(attackers_white) - len(attackers_black))
    return score

def threat_score(board):
    """Evaluate hanging pieces and threats."""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(not piece.color, square)
            defenders = board.attackers(piece.color, square)
            if attackers and not defenders:
                penalty = PIECE_VALUES[piece.piece_type] // 2
                score += -penalty if piece.color == chess.WHITE else penalty
    return score

def bishop_pair_bonus(board):
    """Bonus for bishop pair."""
    white_bishops = sum(1 for sq in board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = sum(1 for sq in board.pieces(chess.BISHOP, chess.BLACK))
    bonus = 30
    return (bonus if white_bishops >= 2 else 0) - (bonus if black_bishops >= 2 else 0)

def evaluate_board(board):
    if board.is_checkmate():
        return -999999 if board.turn else 999999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    endgame = is_endgame(board)

    score += material_score(board)
    score += bishop_pair_bonus(board)
    score += mobility_score(board)
    score += center_control_score(board)
    score += threat_score(board)

    score += king_safety_score(board, chess.WHITE)
    score -= king_safety_score(board, chess.BLACK)

    return score

class CharGPTCodeEvaluator:
    def evaluate_position(self, board: chess.Board) -> float:
        return evaluate_board(board)

    def top_k_moves(self, board, k):
        raise NotImplementedError("Top-k moves not implemented in this evaluator.")


if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    print("Crazy strong evaluation:", evaluate_board(board))
