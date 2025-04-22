import chess
from typing import Optional


# Remove '+' signs and other symbols from a move string
def remove_signs(move_str):
    return move_str.replace('+', '').replace('#', '').replace('x', '')\
        .replace(':', '').replace('-', '')


def match_move(board: chess.Board, move_str: str) -> Optional[chess.Move]:
    legal_moves = board.legal_moves
    # Remove leading ... (seen in gpt-4.1-nano)
    if move_str.startswith('...'):
        move_str = move_str[3:]
    # Check if the move_str matches any legal move in UCI format
    if len(move_str) == 5 and move_str[0] in 'NBRQK' \
            and move_str[1] in 'abcdefgh' and move_str[3] in 'abcdefgh':
        move_str = move_str[1:]
    if len(move_str) == 6 and move_str[0] in 'NBRQK' and move_str[3] in ['-', 'x']:
        move_str = move_str[1:].replace('-', '').replace('x', '')
    for move in legal_moves:
        if move_str == move.uci():
            return move
    # Check if the move_str matches any legal move in SAN format
    for move in legal_moves:
        if remove_signs(board.san(move)) == remove_signs(move_str):
            return move
    return None


def main():
    board = chess.Board('4r1k1/p5p1/1p2q2p/1Qp2p2/3bN2N/1P3KP1/P6P/5B2 b - - 0 1')
    move_str = 'Qxe4#'
    move = match_move(board, move_str)
    print(move)

if __name__ == "__main__":
    main()
