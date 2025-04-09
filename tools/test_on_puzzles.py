import os
import sys
import chess
import chess.engine
import csv
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def download_puzzles(path):
    # Function to download puzzles from the website
    if not os.path.exists(path):
        try:
            import requests
            import zstandard
        except ImportError:
            print("Please install the required libraries: requests, zstandard")
            exit(1)
        url = 'https://database.lichess.org/lichess_db_puzzle.csv.zst'
        response = requests.get(url, stream=True)
        response.raise_for_status()

        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(response.raw) as reader:
            content = reader.read()

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['PuzzleId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'GameUrl'])
            for line in content.split(b'\n')[1:]:
                decoded_line = line.decode('utf-8')
                fields = decoded_line.strip().split(',')
                if len(fields) >= 5 and fields[3].isdigit() and int(fields[3]) < 300:  # Filter easy puzzles
                    writer.writerow(fields)


def test_engine(engine: chess.engine.SimpleEngine, puzzle_csv_path: str, think_time: float):
    stats = {
        'success': 0,
        'failure': 0,
        'total': 0,
        'success_rate': 0.0
    }
    with open(puzzle_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            fen = row['FEN']
            moves = row['Moves'].split()
            pre_move, move = moves[0], moves[1]
            board = chess.Board(fen)
            # First move is before the puzzle position
            board.push(chess.Move.from_uci(pre_move))

            engine_move = engine.play(board, chess.engine.Limit(time=think_time))
            if engine_move.move.uci() == move:
                stats['success'] += 1
            else:
                stats['failure'] += 1
            stats['total'] += 1
            if stats['total'] >= 20:
                break
    stats['success_rate'] = stats['success'] / stats['total'] * 100
    return stats


def main():
    path = 'easy_puzzles.csv'
    download_puzzles(path)
    engine = chess.engine.SimpleEngine.popen_uci([
        sys.executable, 'chessian/main.py', 'mcts'
    ])
    for think_time in [0.1, 0.5, 1.0]:
        print(f"Think time: {think_time} seconds")
        stats = test_engine(engine, path, think_time)
        print(stats)
        print()
    print(stats)
    engine.quit()


if __name__ == "__main__":
    main()
