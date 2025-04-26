import chess
import json
from typing import Optional
from functools import lru_cache
import random
import os
import time
import sys
from chessian.utils import match_move
from chessian.position_encoding import reddit_bot_encoding
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install the required libraries: google-genai")
    exit(1)


class GeminiHeuristic:
    """
    Heuristic based on Google Gemini model
    """
    def __init__(self, model_name, fallback_eval=None):
        """
        Initialize the evaluator with the specified model and device.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.fallback_eval = fallback_eval

    @lru_cache(maxsize=128)
    def call_gemini(self, board_fen: str, k: int) -> dict:
        board = chess.Board(board_fen)
        side_to_move_str = "White" if board.turn == chess.WHITE else "Black"
        moves = [board.san(m) for m in board.legal_moves]
        random.shuffle(moves)
        prompt = ''
        prompt += "You are the best chess player in the world. Please help me with that position.\n"
        prompt += "I want you to rank the position from 0 to 1 (0 is white losing and 1 is white winning).\n"
        prompt += f"Also give me the best {k} moves for {side_to_move_str}.\n"
        prompt += 'The result MUST end with JSON in format \{"score": <int>, "best_moves": [<string>]\}.\n'
        prompt += 'State the check threats and legal moves.\n'
        prompt += '\n'
        prompt += 'The position is:' + board.fen() + '\n'
        prompt += f'Select from the legal moves: ' + ', '.join(moves) + '\n'
        total_slept_time = 0
        for i in range(4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                    ),
                )
                break
            except genai.errors.ClientError as e:
                if i < 3:
                    retry_time = 5 * (2 ** i)
                    for d in e.details['error']['details']:
                        if d['@type'] == 'type.googleapis.com/google.rpc.RetryInfo':
                            retry_time_str = d['retryDelay']
                            retry_time = int(retry_time_str.split('s')[0]) + 3
                    time.sleep(retry_time)
                    total_slept_time += retry_time
                    continue
                else:
                    raise ValueError(f"Error calling Gemini API, i={i}, slept {total_slept_time} seconds.") from e
        # print(response.text)
        json_text = response.text.strip()
        # Find JSON content between curly braces
        start = json_text.find('{')
        end = json_text.rfind('}') + 1
        if start >= 0 and end > 0:
            json_text = json_text[start:end]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {}

    def evaluate_position(self, board: chess.Board) -> float:
        response_dict = self.call_gemini(board.fen(), 5)
        score = response_dict.get('score', 0.0)
        return score

    def top_k_moves(self, board: chess.Board, k: int) -> list:
        result_dict = self.call_gemini(board.fen(), k)
        best_move_str = result_dict.get('best_moves', [])
        best_moves = []
        for move_str in best_move_str:
            move = match_move(board, move_str)
            if move is not None:
                best_moves.append(move)
            else:
                print(f"Invalid move: {move_str} in position {board.fen()}")
        return best_moves[:k]


def main():
    import time
    board = chess.Board('r3k1n1/p5b1/N1p5/3pp1NB/7p/4P2P/2PqQPP1/3R1R1K b - - 1 31')
    print(reddit_bot_encoding(board))
    print('--------------')
    start_time = time.time()
    eval = GeminiHeuristic('gemini-2.0-flash')
    s = eval.evaluate_position(board)
    print(s)
    m = eval.top_k_moves(board, 5)
    print(m)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
