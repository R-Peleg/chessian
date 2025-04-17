import chess
import json
from functools import lru_cache
import random
import os
import time
try:
    from google import genai
except ImportError:
    print("Please install the required libraries: google-genai")
    exit(1)


class GeminiEvaluator:
    """
    Evaluator based on Google Gemini model
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
    
    @staticmethod
    def board_to_string(board: chess.Board) -> str:
        """
        Convert the chess board to a string representation.
        """
        description = ''
        for color in [chess.WHITE, chess.BLACK]:
            color_str = 'white' if color == chess.WHITE else 'black'
            for piece_type in chess.PIECE_TYPES:
                pieces = board.pieces(piece_type, color)
                if not pieces:
                    continue
                squares_str = ', '.join([chess.square_name(sq) for sq in pieces])
                description += f'{color_str} {chess.piece_name(piece_type)}: {squares_str}\n'
        description += f'{"white" if board.turn else "black"} to play\n'
        description += f'Castling: {board.castling_rights}\n'
        description += f'En Passant: {chess.square_name(board.ep_square) if board.ep_square else "None"}\n'
        return description.strip()
    
    @lru_cache(maxsize=128)
    def call_gemini(self, board_fen: str) -> dict:
        board = chess.Board(board_fen)
        prompt = ''
        prompt += "You are a chess engine. You are given a position and you need to evaluate it.\n"
        prompt += 'the result MUST end with JSON in format \{"score": <int>, "best_move": <string>\}.\n'
        prompt += f'example:\n'
        prompt += GeminiEvaluator.board_to_string(chess.Board())
        prompt += "\nEvaluate the position and give a score from 0 to 1, where 0 is a losing position for white and 1 is a winning position for white.\n"
        prompt += '{"score": 0.51, "best_move": "e2e4"}\n\n'
        prompt += GeminiEvaluator.board_to_string(board)
        prompt += "\nEvaluate the position and give a score from 0 to 1, where 0 is a losing position for white and 1 is a winning position for white.\n"
        total_slept_time = 0
        for i in range(4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
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
        response_dict = self.call_gemini(board.fen())
        score = response_dict.get('score', 0.0)
        return score

    def sort_moves(self, board: chess.Board, moves) -> list:
        moves_copy = list(moves)
        random.shuffle(moves_copy)
        result_dict = self.call_gemini(board.fen())
        best_move_str = result_dict.get('best_move')
        best_move_uci = [m for m in moves_copy if m.uci() == best_move_str
                         or board.san(m) == best_move_str]
        if best_move_uci:
            best_move = best_move_uci[0]
            moves_copy.remove(best_move)
            moves_copy.insert(0, best_move)
        return moves_copy


def main():
    import time
    board = chess.Board('8/6k1/2R4p/5p1P/5P1K/6P1/8/r7 b - - 2 58')
    print(GeminiEvaluator.board_to_string(board))
    print('--------------')
    start_time = time.time()
    s = GeminiEvaluator('gemini-2.0-flash').evaluate_position(board)
    print(s)
    m = GeminiEvaluator('gemini-2.0-flash').sort_moves(board, board.legal_moves)
    print(m)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
