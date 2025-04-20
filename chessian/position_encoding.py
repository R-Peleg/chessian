import chess


def reddit_bot_encoding(position: chess.Board) -> str:
    fen_str = position.fen()
    side_to_move_str = "White" if position.turn == chess.WHITE else "Black"
    return f"""
I analyzed the image and this is what I see. Open an appropriate link below and explore the position yourself or with the engine:

{side_to_move_str} to play: <a href="https://chess.com/analysis?fen={fen_str.replace(" ", "+")}">chess.com</a> | <a href="https://lichess.org/analysis/{fen_str.replace(" ", "_")}?color=white">lichess.org</a>
"""
