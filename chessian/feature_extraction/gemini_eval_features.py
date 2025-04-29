"""
Feature extraction from Gemini model evaluation.
"""
import chess
import json
from functools import lru_cache
import random
import os
import time
from chessian.utils import match_move
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install the required libraries: google-genai")
    exit(1)

class GeminiEvalFeatures:
    """
    Class to extract features from Gemini model evaluation.
    """
    def __init__(self, model_name: str) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @lru_cache(maxsize=128)
    def call_gemini(self, board_fen: str) -> dict:
        board = chess.Board(board_fen)
        prompt = f"""
You are an expert chess analyst.
Given the following position FEN: {board_fen}
Evaluate it, write your evaluation with the following JSON:
{{"material": <float>, "pawn_structur": <float>, "mobility": <float>, 'king_safety': <float>}}
Where:
Material represents the piece count advantage for white
Pawns represents the pawn structure advantage for white
Mobility represents the piece development and mobility advantage for white
King Safety represents the king safety advantage for white
Advantage for black is respresented by negative score.
"""
        
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

        json_text = response.text.strip()
        start = json_text.find('{')
        end = json_text.rfind('}') + 1
        if start >= 0 and end > 0:
            json_text = json_text[start:end]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {}

    def get_features(self, board: chess.Board) -> dict[str, float]:
        """
        Extract features from the board using Gemini evaluation.
        """
        response_dict = self.call_gemini(board.fen(), 5)
        return response_dict

def main():
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'gemma-2-27b-it'
    fen = 'rnbq1rk1/1pppppbp/p5p1/7n/P2P4/6P1/1PPNPPBP/RNBQK2R w KQ - 0 7'
    board = chess.Board(fen)
    gemini_eval_features = GeminiEvalFeatures(model_name)
    features = gemini_eval_features.get_features(board)
    print(features)

if __name__ == "__main__":
    main()
