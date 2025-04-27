"""
Dataset creation script.
Dataset from https://theweekinchess.com/assets/files/pgn/wbliw24.pgn is uploaded to

"""

import chess
import chess.pgn
import chess.engine
import pandas as pd
import random
from tqdm import tqdm
import argparse
import uuid


def get_random_moves(board, num_moves):
    """Apply random legal moves to a chess board."""
    moves_applied = 0
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        random_move = random.choice(legal_moves)
        board.push(random_move)
        moves_applied += 1
    return moves_applied

def evaluate_position(engine, board, depth=19):
    """Evaluate a position using Stockfish at specified depth."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].relative.score(mate_score=10000)
        best_move = info["pv"][0] if "pv" in info and info["pv"] else None
        return score, best_move
    except Exception as e:
        print(f"Error evaluating position: {e}")
        return None, None

def get_move_characteristics(board, move):
    """Extract characteristics of a move: capture, check, promotion."""
    if move is None:
        return False, False, False
    
    # Check if move is a capture
    is_capture = board.is_capture(move)
    
    # Create a copy of the board and apply the move to check for check
    temp_board = board.copy()
    temp_board.push(move)
    is_check = temp_board.is_check()
    
    # Check if move is a promotion
    is_promotion = move.promotion is not None
    
    return is_capture, is_check, is_promotion

def process_game(game, engine, game_id, max_random_moves=5, sample_probability=0.05):
    """Process a single chess game and extract position evaluations."""
    data = []
    
    # Skip games without moves
    if not game.headers.get("Result") or not list(game.mainline_moves()):
        return data
    
    board = game.board()
    move_number = 0
    position_count = 0
    
    # Process each position in the game
    for move in game.mainline_moves():
        board.push(move)
        move_number += 1
        
        # Only sample some positions and avoid dataset bias toward common openings
        if move_number < 5:
            continue
        if random.random() > sample_probability:
            continue
            
        # Evaluate original position
        score, best_move = evaluate_position(engine, board)
        if score is not None:
            position_id = f"{game_id}_{position_count}"
            position_count += 1
            
            is_capture, is_check, is_promotion = get_move_characteristics(board, best_move)
            
            data.append({
                'game_id': game_id,
                'position_id': position_id,
                'fen': board.fen(),
                'evaluation': score,
                'best_move': best_move.uci() if best_move else None,
                'bestmove_is_capture': is_capture,
                'bestmove_is_check': is_check,
                'bestmove_is_promotion': is_promotion,
                'random_plies': 0, 
                'moves_from_start': move_number
            })
        
        # Generate random move positions
        if random.random() < 0.3:  # Only do this for some positions
            random_plies = random.randint(1, max_random_moves)
            temp_board = board.copy()
            plies_applied = get_random_moves(temp_board, random_plies)
            
            if plies_applied > 0:
                score, best_move = evaluate_position(engine, temp_board)
                if score is not None:
                    rand_position_id = f"{game_id}_{position_count}_rand{plies_applied}"
                    position_count += 1
                    
                    is_capture, is_check, is_promotion = get_move_characteristics(temp_board, best_move)
                    
                    data.append({
                        'game_id': game_id,
                        'position_id': rand_position_id,
                        'fen': temp_board.fen(),
                        'evaluation': score,
                        'best_move': best_move.uci() if best_move else None,
                        'bestmove_is_capture': is_capture,
                        'bestmove_is_check': is_check,
                        'bestmove_is_promotion': is_promotion,
                        'random_plies': plies_applied,
                        'moves_from_start': move_number + plies_applied
                    })
    
    return data

def create_chess_dataset(pgn_file, stockfish_path, output_file="chess_positions.csv", 
                        max_games=None, max_random_moves=5):
    """Create a dataset of chess position evaluations from a PGN file."""
    # Initialize Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    all_data = []
    games_processed = 0
    
    # Open PGN file
    with open(pgn_file) as pgn:
        # Process each game
        pbar = tqdm(desc="Processing games", unit=" game")
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            # Generate a unique game ID
            game_id = str(uuid.uuid4())[:8]
            
            game_data = process_game(game, engine, game_id, max_random_moves)
            all_data.extend(game_data)
            
            games_processed += 1
            pbar.update(1)
            pbar.set_postfix(positions=len(all_data))
            
            if max_games and games_processed >= max_games:
                break
    
    # Close the engine
    engine.quit()
    pbar.close()
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Dataset created with {len(df)} positions from {games_processed} games")
        return df
    else:
        print("No data collected. Check your PGN file and Stockfish path.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Create a chess position evaluation dataset from PGN files')
    parser.add_argument('pgn_file', help='Path to the PGN file')
    parser.add_argument('stockfish_path', help='Path to the Stockfish executable')
    parser.add_argument('--output', default='chess_positions.csv', help='Output CSV file path')
    parser.add_argument('--max_games', type=int, default=None, help='Maximum number of games to process')
    parser.add_argument('--max_random_moves', type=int, default=5, help='Maximum number of random moves to apply')
    
    args = parser.parse_args()
    
    create_chess_dataset(
        args.pgn_file,
        args.stockfish_path,
        args.output,
        args.max_games,
        args.max_random_moves
    )


if __name__ == "__main__":
    main()
