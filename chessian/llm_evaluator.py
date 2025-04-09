from transformers import AutoModelForCausalLM, AutoTokenizer
import chess


class LLMEvaluator:
    def __init__(self, model_name, device="cuda"):
        """
        Initialize the evaluator with the specified model and device.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
    
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
        prompt = LLMEvaluator.board_to_string(board)
        prompt += "\nLet's have a look on the position and have a quick assessment."
        prompt += "\nWhat the best move here is clearly"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs['input_ids'], max_length=500)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

    def sort_moves(self, board: chess.Board, moves) -> list:
        return moves
    

def main():
    board = chess.Board('8/6k1/2R4p/5p1P/5P1K/6P1/8/r7 w - - 2 58')
    print(LLMEvaluator.board_to_string(board))
    LLMEvaluator('Qwen/Qwen2.5-0.5B', 'cpu').evaluate_position(board)


if __name__ == "__main__":
    main()
