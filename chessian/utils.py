import chess
from typing import Optional


# Remove '+' signs and other symbols from a move string
def remove_signs(move_str):
    return move_str.replace('+', '').replace('#', '').replace('x', '')\
        .replace(':', '').replace('-', '')


def match_move(board: chess.Board, move_str: str) -> Optional[chess.Move]:
    legal_moves = board.legal_moves
    # Check if the move_str matches any legal move in UCI format
    if len(move_str) == 5 and move_str[0] in 'NBRQK':
        move_str = move_str[1:]
    if len(move_str) == 6 and move_str[0] in 'NBRQK' and move_str[3] == '-':
        move_str = move_str[1:].replace('-', '')
    for move in legal_moves:
        if move_str == move.uci():
            return move
    # Check if the move_str matches any legal move in SAN format
    for move in legal_moves:
        if remove_signs(board.san(move)) == remove_signs(move_str):
            return move
    return None
