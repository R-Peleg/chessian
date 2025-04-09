import sys
import chess
import chess.pgn
import chess.engine

def main():
    # Initialize two UCI engines
    engine1 = chess.engine.SimpleEngine.popen_uci([
        sys.executable, 'chessian/main.py', 'mcts'
    ])
    engine2 = chess.engine.SimpleEngine.popen_uci([
        sys.executable, 'chessian/main.py', 'mcts'
    ])

    board = chess.Board()

    while not board.is_game_over():
        # Alternate between engines
        current_engine = engine1 if board.turn == chess.WHITE else engine2
        result = current_engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        print(board)
        print()
        if len(board.move_stack) > 100:
            print("Game over due to 100 moves timeout.")
            break

    # Print the game PGN
    game = chess.pgn.Game.from_board(board)
    print(game)

    # Cleanup
    engine1.quit()
    engine2.quit()

if __name__ == "__main__":
    main()
