import argparse
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import chess

def main():
    parser = argparse.ArgumentParser(description='Run Lasso regression on joined dataset and features.')
    parser.add_argument('--dataset', required=True, help='Hugging Face dataset name (e.g., username/dataset)')
    parser.add_argument('--features', required=True, help='CSV file with extracted features')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    args = parser.parse_args([
        '--dataset', 'ReuvenP/chess_eval_for_search',
        '--feature', 'classic_features.csv'
    ])

    from datasets import load_dataset

    dataset = load_dataset(args.dataset)
    if 'train' in dataset:
        df = pd.DataFrame(dataset['train'])
    else:
        default_split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[default_split])

    features_df = pd.read_csv(args.features)

    # Filter out positions where the side to move is in check
    def not_in_check(fen):
        try:
            board = chess.Board(fen)
            return not board.is_check()
        except Exception:
            return False
    df = df[df['fen'].apply(not_in_check)]

    # Flip evaluation if black to move, since eval is relative
    def flip_eval_if_black(row):
        fen = row['fen']
        eval_value = row['evaluation']
        try:
            board = chess.Board(fen)
            if board.turn == chess.BLACK:
                return -eval_value
            else:
                return eval_value
        except Exception:
            return eval_value
    df['evaluation'] = df.apply(flip_eval_if_black, axis=1)

    # Filter out evaluation > 500 or < -500
    df = df[(df['evaluation'] <= 500) & (df['evaluation'] >= -500)]

    # Join on position_id
    merged = pd.merge(df, features_df, on='position_id')

    # Use 'evaluation' as the target column
    target = merged['evaluation']
    feature_cols = [col for col in features_df.columns if col != 'position_id']
    X = merged[feature_cols] * 100
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    lasso = Lasso(alpha=1)
    lasso.fit(X_train, y_train)

    print('Feature weights:')
    for name, coef in zip(feature_cols, lasso.coef_):
        print(f'{name}: {coef}')

    preds = lasso.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f'Mean Squared Error (Lasso): {mse}')

    # Regular linear regression with only Material feature
    from sklearn.linear_model import LinearRegression
    if 'Material' in feature_cols:
        X_material = merged[['Material']] * 100
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_material, y, test_size=args.test_size, random_state=42)
        lr = LinearRegression()
        lr.fit(X_train_m, y_train_m)
        preds_m = lr.predict(X_test_m)
        mse_m = mean_squared_error(y_test_m, preds_m)
        print(f'Mean Squared Error (Linear Regression, Material only): {mse_m}')
    else:
        print('Material feature not found in features.')

if __name__ == "__main__":
    main()
