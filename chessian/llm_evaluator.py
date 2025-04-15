from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import re


class LLMEvaluator:
    def __init__(self, model_name, device="cuda", fallback_eval=None):
        """
        Initialize the evaluator with the specified model and device.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
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

    def evaluate_position(self, board: chess.Board) -> float:
        starting_position = chess.Board()
        prompt = ''
        prompt += "You are a chess engine. You are given a position and you need to evaluate it.\n"
        prompt += f'example:\n'
        prompt += LLMEvaluator.board_to_string(starting_position)
        prompt += "\nEvaluate the position and give a score from -1 to 1, where -1 is a losing position for white and 1 is a winning position for white.\n"
        prompt += "Score: 0.12\n\n"
        prompt += LLMEvaluator.board_to_string(board)
        prompt += "\nEvaluate the position and give a score from -1 to 1, where -1 is a losing position for white and 1 is a winning position for white.\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs['input_ids'],
                                      attention_mask=inputs['attention_mask'],
                                      do_sample=False,
                                      top_k=None, top_p=None,
                                      max_new_tokens=7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_score = re.search(r'Score: (-?\d+\.\d+)', response)
        if response_score:
            score = float(response_score.group(1))
            return score
        else:
            return 0.0

    def sort_moves(self, board: chess.Board, moves) -> list:
        if self.fallback_eval:
            return self.fallback_eval.sort_moves(board, moves)
        else:
            return moves


def main():
    import time
    board = chess.Board('8/6k1/2R4p/5p1P/5P1K/6P1/8/r7 w - - 2 58')
    print(LLMEvaluator.board_to_string(board))
    start_time = time.time()
    LLMEvaluator('Qwen/Qwen2.5-0.5B-Instruct', 'cpu').evaluate_position(board)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
