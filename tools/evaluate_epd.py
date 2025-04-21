"""
Evaluate heuristic on EPD chess position file.
"""
import chess
from chessian.gemini_heuristic import GeminiHeuristic


def test_heuristic(heuristic, epd_file_path):
    stats = {
        'top1': 0,
        'top3': 0,
        'top5': 0,
        'failure': 0,
        'no_moves': 0,
        'total': 0,
    }
    with open(epd_file_path, 'r') as epd_file:
        for line in epd_file:
            if line.startswith('#'):
                continue
            # Parse EPD line
            parts = line.strip().split(';')
            fen_and_best_move = parts[0].strip()
            # Format like 'rnbqk1nr/ppp2ppp/4p3/3p4/3PP3/b1N5/PPP2PPP/R1BQKBNR w KQkq - bm Bxa3'
            fen, best_move_str = fen_and_best_move.split(' bm ')
            board = chess.Board(fen)
            best_moves = [board.parse_san(m) for m in best_move_str.split(' ')]

            heuristic_moves = heuristic.top_k_moves(board, 5)
            stats['total'] += 1
            if not heuristic_moves:
                stats['no_moves'] += 1
                continue
            # Check if best move is in the top 5 moves
            if set(best_moves).intersection(set(heuristic_moves[:1])):
                stats['top1'] += 1
            if set(best_moves).intersection(set(heuristic_moves[:3])):
                stats['top3'] += 1
            if set(best_moves).intersection(set(heuristic_moves[:5])):
                stats['top5'] += 1
            else:
                stats['failure'] += 1
    return stats
            

def main():
    # TODO: argparse
    stats = test_heuristic(GeminiHeuristic('gemini-2.0-flash'), 'tools/LVL0.epd')
    print(stats)


if __name__ == "__main__":
    main()
