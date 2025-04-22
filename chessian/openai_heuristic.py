import chess
import json
from typing import Optional
from functools import lru_cache
import random
import os
import time
import sys
from chessian.position_encoding import reddit_bot_encoding
from chessian.utils import match_move
from chessian.llm_evaluator import LLMEvaluator
try:
    import openai
except ImportError:
    print("Please install the required libraries: openai")
    exit(1)


class OpenAIHeuristic:
    """
    Heuristic based on OpenAI model
    """
    def __init__(self, model_name, fallback_eval=None):
        """
        Initialize the evaluator with the specified model and device.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.fallback_eval = fallback_eval

    @lru_cache(maxsize=128)
    def call_openai(self, board_fen: str, k: int) -> dict:
        board = chess.Board(board_fen)
        side_to_move_str = "White" if board.turn == chess.WHITE else "Black"
        # moves = [board.san(m) for m in board.legal_moves]
        # random.shuffle(moves)
        move_list_format = "[" + ', '.join(['"<move' + str(i+1) + '>"' for i in range(k)]) + "]"
        prompt = (
            'You are the world\'s strongest chess engine. \n'
            'Given the FEN below, evaluate the position **from White\'s perspective** on a scale from -10 to +10:\n'
            '-10 = completely losing, 0 = equal position, +10 = completely winning.\n'
            '\n'
            f'Then, suggest the **{k} best moves for White**, in algebraic notation (e.g., "Nf3", "dxe5").\n'
            '\n'
            'Think step by step and explain your reasoning. Count pieces, check for develoment, and assess king safety.\n'
            'Your final response MUST be a valid JSON object in this exact format:'
            f'{{"score": <integer between -10 and 10>, "best_moves": {move_list_format}}}\n'
            '\n'
            f'FEN: {board_fen}\n'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error calling OpenAI API: {str(e)}")
        print(response_text)
        json_text = response_text.strip()
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
        response_dict = self.call_openai(board.fen(), 5)
        score = response_dict.get('score', 0.0)
        return score

    def top_k_moves(self, board: chess.Board, k: int) -> list:
        result_dict = self.call_openai(board.fen(), k)
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
    board = chess.Board('rnbqk1nr/pp1p1ppp/4p3/2b5/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 2')
    print('--------------')
    start_time = time.time()
    eval = OpenAIHeuristic('gpt-4.1')
    s = eval.evaluate_position(board)
    print(s)
    m = eval.top_k_moves(board, 5)
    print(m)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
